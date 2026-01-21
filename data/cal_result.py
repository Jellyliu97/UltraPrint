import os

def calculate_metrics(root_path, is_true_positive=True):
    """
    Calculates accuracy and recall for positive or negative samples in a nested directory structure.
    
    Args:
        root_path (str): The root directory containing the test dataset.
    """
    total_samples = 0
    true_positives = 0
    true_negatives = 0

    # choose_cal_user_lists = ['LGD_a4', 'lib_a2', 'lib_a4', 'lib1'] #设备自身震动伪造的数据

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_path):
        # if any(user in dirpath for user in choose_cal_user_lists):
            for filename in filenames:
                if filename.endswith('.txt'):
                # if filename.endswith('.txt') and ('fake' in filename):
                    file_path = os.path.join(dirpath, filename)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            # Assuming the file contains a single float number
                            score = float(content)
                            
                            # # Determine prediction: 1 if score > 0 else 0
                            # prediction = 1 if score > 0 else 0
                            
                            total_samples += 1
                            
                            # Prediction logic: score > 0 -> 1 (Positive), else -> 0 (Negative)
                            # Since all samples in this path are Ground Truth Positive:
                            if score > 0:
                                true_positives += 1
                            else:
                                true_negatives += 1
                            
                                
                    except ValueError:
                        print(f"Warning: Could not parse number in file {file_path}")
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

    # Calculate metrics
    # Since all samples are positive, Accuracy = Recall = TP / Total Positive Samples
    if total_samples == 0:
        print("No .txt files found.")
        return

    if is_true_positive:
        accuracy = true_positives / total_samples
        recall = true_positives / total_samples

        print(f"Total Samples (All Positive): {total_samples}")
        print(f"True Positives (Predicted > 0): {true_positives}")
        print("-" * 30)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall:   {recall:.4f}")
    else:
        accuracy = true_negatives / total_samples
        recall = true_negatives / total_samples

        print(f"Total Samples (All Negative): {total_samples}")
        print(f"True Negatives (Predicted <= 0): {true_negatives}")
        print("-" * 30)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall:   {recall:.4f}")

if __name__ == "__main__":
    # Note: Using raw string (r"...") to handle backslashes in Windows paths correctly

    dataset_path = r"E:\dataset\ultrasound_video_audio\record\stage2_test_dataset_ldxTest"
    # dataset_path = r"E:\dataset\ultrasound_video_audio\record\stage2_test_dataset_fakeMismatch"

    
    if os.path.exists(dataset_path):
        # calculate_metrics(dataset_path, is_true_positive=False)
        calculate_metrics(dataset_path, is_true_positive=True)
    else:
        print(f"Error: The path '{dataset_path}' does not exist.")