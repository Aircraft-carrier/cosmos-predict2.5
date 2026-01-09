from lerobot.datasets.lerobot_dataset import LeRobotDataset


dataset_root = "/gemini/platform/public/embodiedAI/users/zsh/dataset/Lerobot/hflibero"

delta_timestamps = {
    "action" : [i / 10 for i in range(17)],
    "observation.images.image" : [i / 10 for i in range(17)] 
}

dataset = LeRobotDataset(repo_id="HuggingFaceVLA/libero", root=dataset_root, delta_timestamps=delta_timestamps)
# print(dataset.delta_indices)

for idx in list(range(1000, len(dataset), 5)):
    data = dataset[idx]
    print(
        (
            f"{idx} : {data.keys()}\n"
            f"{data['observation.images.image'].shape}\n"
            f"{data['action'].shape}\n"
            "---"
        )
    )