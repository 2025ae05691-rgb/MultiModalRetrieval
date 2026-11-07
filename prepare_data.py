import argparse
from data_utils import prepare_subset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 100 MSR-VTT videos and 200 captions.")
    parser.add_argument("--json", default="data/annotations/MSRVTT_100.json", help="Optional path to MSR-VTT annotation JSON.", type=str)
    parser.add_argument("--videos-per-category", type=int, default=5)
    parser.add_argument("--captions-per-video", type=int, default=2)
    parser.add_argument("--target-videos", type=int, default=100)
    args = parser.parse_args()
    prepare_subset(
        target_videos=args.target_videos,
        videos_per_category=args.videos_per_category,
        captions_per_video=args.captions_per_video,
        json_path=args.json
    )
