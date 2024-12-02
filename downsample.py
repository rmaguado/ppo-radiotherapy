import trimesh


def main(model_dir: str, output_dir: str, target_face_count: int, aggression: float):

    mesh = trimesh.load(model_dir)

    # print(f"Initial number of faces: {len(mesh.faces)}")

    downsampled_mesh = mesh.simplify_quadric_decimation(
        face_count=target_face_count, aggression=aggression
    )

    print(f"Number of faces after downsampling: {len(downsampled_mesh.faces)}")

    downsampled_mesh.export(output_dir)


if __name__ == "__main__":
    model_dir = "./models/segmentation.obj"
    output_dir = "./models/downsampled/segmentation.obj"
    target_face_count = 1000
    aggression = 5

    main(model_dir, output_dir, target_face_count, aggression)
