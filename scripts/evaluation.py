import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config_utils import get_path, get_setting, load_project_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate trained image classifier.")
	parser.add_argument("--config", type=Path, default=Path("config/project_config.json"))
	parser.add_argument("--processed-dir", type=Path, default=None)
	parser.add_argument("--model-path", type=Path, default=None)
	parser.add_argument("--models-dir", type=Path, default=None)
	parser.add_argument("--figures-dir", type=Path, default=None)
	parser.add_argument("--img-size", type=int, default=None)
	parser.add_argument("--batch-size", type=int, default=None)
	return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
	config = load_project_config(args.config)

	args.processed_dir = args.processed_dir or get_path(config, "processed_dir", "data/processed")
	args.models_dir = args.models_dir or get_path(config, "models_dir", "models")
	args.figures_dir = args.figures_dir or get_path(config, "figures_dir", "Figures")
	args.model_path = args.model_path or Path(
		get_setting(config, "evaluation", "model_path", "models/trained_model.h5")
	)
	args.img_size = args.img_size or int(get_setting(config, "evaluation", "img_size", 224))
	args.batch_size = args.batch_size or int(get_setting(config, "evaluation", "batch_size", 16))
	return args


def save_confusion_matrix_plot(cm: np.ndarray, labels: list, out_path: Path) -> None:
	out_path.parent.mkdir(parents=True, exist_ok=True)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
	plt.title("Confusion Matrix")
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.tight_layout()
	plt.savefig(out_path, dpi=150)
	plt.close()


def main() -> None:
	args = resolve_settings(parse_args())

	val_dir = args.processed_dir / "val"
	if not val_dir.exists():
		raise FileNotFoundError("Validation directory not found. Run preprocessing first.")
	if not args.model_path.exists():
		raise FileNotFoundError(f"Model file not found: {args.model_path}")

	args.models_dir.mkdir(parents=True, exist_ok=True)
	args.figures_dir.mkdir(parents=True, exist_ok=True)

	val_gen = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(
		str(val_dir),
		target_size=(args.img_size, args.img_size),
		batch_size=args.batch_size,
		class_mode="categorical",
		shuffle=False,
	)

	model = tf.keras.models.load_model(args.model_path)

	pred_probs = model.predict(val_gen, verbose=1)
	pred_labels = np.argmax(pred_probs, axis=1)
	true_labels = val_gen.classes

	class_names = [name for name, _ in sorted(val_gen.class_indices.items(), key=lambda x: x[1])]

	cm = confusion_matrix(true_labels, pred_labels)
	report_dict = classification_report(
		true_labels,
		pred_labels,
		target_names=class_names,
		output_dict=True,
		zero_division=0,
	)

	cm_path = args.figures_dir / "confusion_matrix.png"
	save_confusion_matrix_plot(cm, class_names, cm_path)

	metrics_path = args.models_dir / "evaluation_metrics.json"
	with metrics_path.open("w", encoding="utf-8") as f:
		json.dump(
			{
				"confusion_matrix": cm.tolist(),
				"classification_report": report_dict,
				"class_indices": val_gen.class_indices,
			},
			f,
			indent=2,
		)

	print(f"Saved confusion matrix plot: {cm_path}")
	print(f"Saved metrics: {metrics_path}")
	print(f"Validation accuracy: {report_dict['accuracy']:.4f}")


if __name__ == "__main__":
	main()
