import numpy as np
from app.third_party.image_stitching.image import Image
from app.third_party.image_stitching.multi_images_matches import MultiImageMatches
from app.third_party.image_stitching.pair_match import PairMatch
from app.third_party.image_stitching.connected_components import find_connected_components
from app.third_party.image_stitching.build_homographies import build_homographies
from app.third_party.image_stitching.inpaint_blending import inpaint_blending

def stitch_inpainting(images: list[np.ndarray]) -> list[np.ndarray]:
    """Stitch images together and inpaint the seams."""
    stitching_images = [Image(image) for image in images]
    for image in stitching_images:
        image.compute_features()
    print("Computing matches...")
    matcher = MultiImageMatches(stitching_images)
    pair_matches: list[PairMatch] = matcher.get_pair_matches()
    pair_matches.sort(key=lambda pair_match: len(pair_match.matches), reverse=True)
    print("Computing connected components...")
    connected_components = find_connected_components(pair_matches)
    print(f"Found {len(connected_components)} connected components.")
    print("Building homographies...")
    build_homographies(connected_components, pair_matches)
    print("Inpainting...")
    results = [
        inpaint_blending(connected_component)
        for connected_component in connected_components
    ]

    return results
