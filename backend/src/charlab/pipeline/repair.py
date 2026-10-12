"""Face repair: redraw a weak face with the generator and blend it back.

The face is cut out as a square 2.2 times its size and enlarged to 1024x1024. The generator
redraws it, with the crop as Picture 1 and, optionally, a face reference as Picture 2. The
result is shrunk back and blended in through a soft oval over the face, so nothing else in
the image changes.

In GPU experiments (2026-10-11/12), this approach restored deliberately blurred faces (face
sharpness 0.01 → 0.60 of the reference's) and sharpened small real faces. Qwen's inpainting
pipeline and an anime SDXL model did neither. It can also replace the pose: with a
front-facing reference it drew a front face over a side profile. So a repair is a candidate
only: it has to be checked before it replaces the original.

The checks do not cover fine facial features: repairs widened the eyes, lightened the irises
and thickened the brows while CCIP still scored the same character. Face repair is therefore
off by default, and is meant for faces that are broken anyway.
"""

import logging

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from charlab.config import FaceRepairSettings
from charlab.pipeline.faces import Face
from charlab.pipeline.jobs import Check, Job, Repair
from charlab.pipeline.quality import QualityGate
from charlab.pipeline.stages import Generate

log = logging.getLogger(__name__)

REPAIR_SIZE = 1024
# The crop's side as a multiple of the face's longer side: room for hair and neck, so the
# redrawn face fits its surroundings.
CROP_SCALE = 2.2
MIN_CROP = 128

# A repair's seed is the kept attempt's seed plus this. Retry seeds step by 1000, so a repair
# never reuses one.
REPAIR_SEED_STEP = 500

PROMPT = (
    "Redraw Picture 1 exactly, with the same framing, head angle, pose, expression, hair and "
    "lighting, but with sharper, cleaner detail in the face and eyes. Keep the art style of "
    "Picture 1."
)
REFERENCE_PROMPT = " Picture 2 shows the same character's face; match the face to it."


def crop_box(face: Face, image: Image.Image) -> tuple[int, int, int, int]:
    """A square around the face, CROP_SCALE times its longer side, moved inside the image."""
    left, top, right, bottom = face.box
    side = max(int(CROP_SCALE * max(right - left, bottom - top)), MIN_CROP)
    side = min(side, image.width, image.height)
    cx, cy = (left + right) // 2, (top + bottom) // 2
    x0 = min(max(cx - side // 2, 0), image.width - side)
    y0 = min(max(cy - side // 2, 0), image.height - side)
    return x0, y0, x0 + side, y0 + side


def oval_mask(
    face: Face, box: tuple[int, int, int, int], grow: float, feather: float
) -> Image.Image:
    """An L mask the size of `box`: a soft oval over the face, enlarged by `grow` of the face's
    size and blurred by `feather` of the box's side."""
    left, top, right, bottom = face.box
    side = box[2] - box[0]
    pad_x, pad_y = grow * (right - left), grow * (bottom - top)
    mask = Image.new("L", (side, side), 0)
    ImageDraw.Draw(mask).ellipse(
        [
            left - pad_x - box[0],
            top - pad_y - box[1],
            right + pad_x - box[0],
            bottom + pad_y - box[1],
        ],
        fill=255,
    )
    return mask.filter(ImageFilter.GaussianBlur(feather * side))


def blend(
    image: Image.Image, face: Face, box: tuple[int, int, int, int], redrawn: Image.Image
) -> Image.Image:
    """Paste the redrawn crop back, through a soft oval over the face only."""
    side = box[2] - box[0]
    patch = redrawn.convert("RGB").resize((side, side), Image.Resampling.LANCZOS)
    out = image.copy()
    out.paste(patch, box[:2], oval_mask(face, box, grow=0.35, feather=0.04))
    return out


class FaceRepairer:
    """Redraws one face with the generate stage, so no extra model is loaded."""

    def __init__(self, generate: Generate):
        self.generate = generate

    def __call__(
        self, image: Image.Image, face: Face, seed: int, reference: Image.Image | None = None
    ) -> Image.Image:
        """The image with the face redrawn. `reference`, a close-up of the character's face,
        becomes Picture 2; without it the generator sees only the crop."""
        box = crop_box(face, image)
        crop = image.crop(box).resize((REPAIR_SIZE, REPAIR_SIZE), Image.Resampling.LANCZOS)
        prompt, inputs = PROMPT, [crop]
        if reference is not None:
            prompt, inputs = PROMPT + REFERENCE_PROMPT, [crop, reference]
        redrawn = self.generate(prompt, inputs, (REPAIR_SIZE, REPAIR_SIZE), seed)
        return blend(image, face, box, redrawn)


# Layout is compared on a 16x16 greyscale thumbnail of the crop: fine detail (what a repair
# should change) is averaged away, while the pose and the face's place (what it must keep)
# remain.
LAYOUT_SIZE = 16


def layout_similarity(a: Image.Image, b: Image.Image) -> float:
    """Correlation of two images' coarse layouts: 1.0 when only fine detail differs."""
    vectors = []
    for image in (a, b):
        grey = image.convert("L").resize((LAYOUT_SIZE,) * 2, Image.Resampling.BOX)
        vector = np.asarray(grey, dtype=np.float32).flatten()
        vector -= vector.mean()
        vectors.append(vector / (np.linalg.norm(vector) or 1.0))
    return float(vectors[0] @ vectors[1])


def _centre(face: Face) -> np.ndarray:
    left, top, right, bottom = face.box
    return np.array([(left + right) / 2, (top + bottom) / 2])


def keep_repair(
    before: Image.Image,
    after: Image.Image,
    before_check: Check,
    after_check: Check,
    limits: FaceRepairSettings,
) -> tuple[bool, str]:
    """Whether the repaired image should replace the original, and why.

    A repair must keep the structure (the same layout, the face in the same place) and be
    better: sharper, no further from the character, and failing no more checks. The structure
    test exists because a better identity score alone can be gamed: in the GPU experiment, a
    front face drawn over a side profile scored closer to the front-facing reference.
    """
    face = before_check.face
    if face is None or after_check.face is None:
        return False, "no face found after the repair"
    box = crop_box(face, before)
    layout = layout_similarity(before.crop(box), after.crop(box))
    if layout < limits.min_layout_similarity:
        return False, f"layout changed: {layout:.2f} < {limits.min_layout_similarity}"
    moved = float(np.linalg.norm(_centre(after_check.face) - _centre(face))) / face.size
    if moved > limits.max_face_shift:
        return False, f"face moved {moved:.2f}x its size, more than {limits.max_face_shift}x"
    sharp_before, sharp_after = before_check.sharpness, after_check.sharpness
    if sharp_before is not None and sharp_after is not None and sharp_after <= sharp_before:
        return False, f"not sharper: {sharp_after:.2f} <= {sharp_before:.2f}"
    worse = after_check.identity - before_check.identity
    if worse > limits.identity_tolerance:
        return False, (
            f"less like the character: CCIP {after_check.identity:.3f} vs "
            f"{before_check.identity:.3f}"
        )
    if len(after_check.failures) > len(before_check.failures):
        return False, "more checks failed after the repair"
    return True, (
        f"kept: layout {layout:.2f}, face moved {moved:.2f}x, sharpness "
        f"{sharp_before or 0:.2f} -> {sharp_after or 0:.2f}"
    )


class FaceRepairStage:
    """The repair stage: picks the faces that need redrawing, repairs them, checks the result
    with the quality gate, and keeps it only if `keep_repair` agrees."""

    def __init__(self, settings: FaceRepairSettings, generate: Generate, gate: QualityGate):
        self.settings = settings
        self.gate = gate
        self.repairer = FaceRepairer(generate)
        self._reference: Image.Image | None = None

    def start(self, reference: Image.Image) -> None:
        """Cut the face reference (Picture 2 of every repair) from the prepared reference."""
        if not self.settings.use_reference:
            return
        faces = self.gate.detect(reference)
        if faces:
            self._reference = reference.crop(crop_box(faces[0], reference))
        else:
            log.warning("no face found in the reference; repairs run without a face reference")

    def needs_repair(self, job: Job, check: Check) -> bool:
        if self.settings.mode == "off" or not job.face or check.face is None:
            return False
        if check.failures:
            return True
        return self.settings.mode == "small" and check.face.size < self.settings.small_below_px

    def __call__(
        self, job: Job, image: Image.Image, check: Check, seed: int
    ) -> tuple[Repair, Image.Image] | None:
        if not self.needs_repair(job, check):
            return None
        repair_seed = seed + REPAIR_SEED_STEP
        repaired = self.repairer(image, check.face, repair_seed, self._reference)
        after = self.gate(job, repaired)
        keep, reason = keep_repair(image, repaired, check, after, self.settings)
        log.info("%s, face repair: %s", job.key, reason)
        return Repair(repair_seed, after, keep, reason), repaired if keep else image
