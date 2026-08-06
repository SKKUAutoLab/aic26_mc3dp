
import time

from . import cli
from .filterer import NoiseFilter

from tqdm import tqdm

def main():
	args = cli.parse_args(prog="run_filter")
	started = time.perf_counter()

	noise_filter = NoiseFilter(
		submission=args.input, dataset=args.dataset, pose_model=args.pose_model,
		gpu=args.gpu, output=args.output, split=args.split, scene=args.scene,
		start=args.start, end=args.end, zone_dir=args.zone_dir)

	total = noise_filter.num_frames
	for k, frame_id in enumerate(tqdm(range(noise_filter.start, noise_filter.end + 1))):
		noise_filter.filter_frame(frame_id)
		# if k and k % 100 == 0:
		#     elapsed = time.perf_counter() - started
		#     rate = elapsed / k
		#     print(f"[{k / total * 100:5.1f}%] frame {frame_id} · {rate:.2f}s/frame · "
		#           f"~{(total - k) * rate / 60:.0f}m left · "
		#           f"kept {noise_filter.stats['kept']}/{noise_filter.stats['persons']}", flush=True)

	stats = noise_filter.close()
	persons = stats["persons"] or 1
	print(f"\nFILTERED -> {stats['output']}")
	print(f"  person rows : {stats['kept']} kept, {stats['dropped']} dropped "
		  f"({stats['dropped'] / persons * 100:.1f}% removed as noise)")
	print(f"  outside every zone (fell back to best cameras): {stats['fallback']}")
	print(f"  no camera could see them at all: {stats['no_view']}")
	print(f"  time: {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
	main()
