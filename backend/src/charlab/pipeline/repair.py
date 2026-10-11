"""Face repair: redraw a weak face with the generator and blend it back.

The face is cut out as a square 2.2 times its size and enlarged to 1024x1024. The generator
redraws it, with the crop as Picture 1 and, optionally, a face reference as Picture 2. The
result is shrunk back and blended in through a soft oval over the face, so nothing else in
the image changes.

In a GPU experiment (2026-10-11), this approach restored deliberately blurred faces (face
sharpness 0.01 → 0.60 of the reference's) and sharpened small real faces. Qwen's inpainting
pipeline and an anime SDXL model did neither. It can also replace the pose: with a
front-facing reference it drew a front face over a side profile. So a repair is a candidate
only: it has to be checked before it replaces the original.
"""

from PIL import Image, ImageDraw, ImageFilter

from charlab.pipeline.faces import Face
from charlab.pipeline.stages import Generate

REPAIR_SIZE = 1024
# The crop's side as a multiple of the face's longer side: room for hair and neck, so the
# redrawn face fits its surroundings.
CROP_SCALE = 2.2
MIN_CROP = 128

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
