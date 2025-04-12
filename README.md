# Vietnamese OCR with Curriculum Learning

This project implements a Vietnamese OCR system with diacritic recognition, enhanced by curriculum learning. The system progressively learns from simpler tasks (character recognition) to more complex ones (full word and sentence recognition with diacritics).

## Features

- **Curriculum Learning**: Trains the model progressively from easier to harder examples
- **Multi-Level Recognition**: Works with characters, words, and sentences
- **Diacritic Handling**: Explicitly models Vietnamese diacritics
- **Base Character Recognition**: Recognizes base characters separately from diacritics
- **Flexible Training**: Supports both standard and curriculum-based training
- **Detailed Analytics**: Provides rich insights into training progress

## How Curriculum Learning Works

Curriculum learning is a training technique where we initially train on simpler examples and gradually introduce more complex ones. For Vietnamese OCR, complexity is defined by:

1. **Text Length**: Single characters → Short words → Long words → Sentences
2. **Diacritic Complexity**: No diacritics → Simple diacritics → Complex diacritic combinations
3. **Combined Complexity**: A weighted combination of the above factors

The training process adapts to the model's progress, advancing to more complex stages when the model plateaus on easier content.

## Project Structure

- `main_with_curriculum.py`: Main script with curriculum learning support
- `curriculum.py`: Core curriculum learning implementation
- `curriculum_trainer.py`: Extended trainer for curriculum learning
- `dataset.py`: Vietnamese character dataset handling
- `models.py`: OCR model architecture with diacritic recognition
- `trainer.py`: Standard training loop
- `utils.py`: Helper functions
- `analyze_dataset.py`: Tool to analyze dataset complexity
- `inference_demo.py`: Demo script for using the trained model

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install torch transformers datasets wandb pillow matplotlib tqdm numpy
```

## Step 1: Analyze the Dataset

Before training, analyze the dataset to understand its complexity distribution:

```bash
python analyze_dataset.py --dataset_name vklinhhh/vietnamese_character_diacritic --output_dir dataset_analysis
```

This will create visualizations and recommendations for curriculum stages in the `dataset_analysis` directory.

## Step 2: Train with Curriculum Learning

Train the model with curriculum learning:

```bash
python main_with_curriculum.py \
  --dataset_name vklinhhh/vietnamese_character_diacritic \
  --use_curriculum \
  --curriculum_strategy combined \
  --curriculum_stages 3 \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 5e-6 \
  --wandb_project vietnamese-ocr-curriculum
```
╰λ python main_with_curriculum.py --use_curriculum --curriculum_strategy combined --curriculum_stages 3 --learning_rate 5e-7 --stage_epochs 2,3,4 --epochs 9

### Key Parameters:

- `--use_curriculum`: Enable curriculum learning
- `--curriculum_strategy`: Choose from 'length', 'complexity', or 'combined'
- `--curriculum_stages`: Number of stages (default: 3)
- `--stage_epochs`: Optionally specify epochs per stage (e.g., "5,5,5")
- `--stage_patience`: How many epochs to wait before advancing stage (default: 3)

## Step 3: Evaluate and Visualize Results

Run inference on the validation set:

```bash
python inference_demo.py \
  --model_dir vietnamese-ocr-curriculum-final \
  --evaluate_dataset \
  --dataset_name vklinhhh/vietnamese_character_diacritic \
  --split validation \
  --num_samples 10
```

Or test on a single image:

```bash
python inference_demo.py \
  --model_dir vietnamese-ocr-curriculum-final \
  --image_path path/to/image.jpg
```

## Understanding Curriculum Stages

The default curriculum has three stages:

1. **Stage 1 (Basic)**: Single characters and simple words with few diacritics
2. **Stage 2 (Intermediate)**: Longer words and more complex diacritics
3. **Stage 3 (Advanced)**: Sentences and challenging diacritic combinations

The model begins training on Stage 1, and automatically advances when performance plateaus, focusing on different aspects at each stage.

## Custom Curriculum Design

You can customize the curriculum by:

1. Analyzing your dataset with `analyze_dataset.py`
2. Adjusting `--curriculum_stages` based on the analysis
3. Choosing a suitable `--curriculum_strategy` ('length', 'complexity', or 'combined')
4. Setting custom `--stage_epochs` or using automatic progression with `--stage_patience`

## Comparing with Standard Training

To compare curriculum learning with standard training:

1. Train with curriculum: Use `--use_curriculum`
2. Train without curriculum: Omit `--use_curriculum`
3. Compare metrics in Weights & Biases dashboard

## Visualization in Weights & Biases

The training process logs detailed metrics to Weights & Biases, including:

- Loss curves for each curriculum stage
- Accuracies: word, character, base characters, and diacritics
- Curriculum progression visualization
- Example predictions with image visualizations

## Future Improvements

- Add support for mixed datasets (characters, words, and sentences)
- Implement more advanced curriculum strategies
- Add additional data augmentation for each curriculum stage
- Explore transfer learning between stages# Vietnamese-TrOCR-Diacritic-Focus
