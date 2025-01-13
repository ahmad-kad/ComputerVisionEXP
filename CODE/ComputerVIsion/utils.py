import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Optional, Dict
import cv2
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import random
from PIL import Image
from typing import Union, List, Dict, Optional, Tuple
from utils import ImagePreprocessor, ImageAnalyzer, ImageAugmentor
import seaborn as sns
import plotly.express as px

   
class ImageLoaderComparator:
    """Class for loading and comparing multiple images"""
    
    def __init__(self, 
                 base_dir: Union[str, Path],
                 preprocessor: Optional[ImagePreprocessor] = None,
                 analyzer: Optional[ImageAnalyzer] = None):
        """
        Initialize with directory path and optional preprocessor/analyzer
        
        Args:
            base_dir: Base directory containing images
            preprocessor: Optional ImagePreprocessor instance
            analyzer: Optional ImageAnalyzer instance
        """
        self.base_dir = Path(base_dir)
        self.image_paths = self._index_images()
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.analyzer = analyzer or ImageAnalyzer()
        self.loaded_images = {}
        self.comparison_results = pd.DataFrame()
        
    def _index_images(self) -> Dict[str, Path]:
        """Index all images in directory and subdirectories"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = {}
        
        for ext in image_extensions:
            for path in self.base_dir.rglob(f'*{ext}'):
                image_paths[path.name] = path
                
        return image_paths
    
    def load_specific_images(self, image_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Load specific images by name
        
        Args:
            image_names: List of image names to load
            
        Returns:
            Dictionary of loaded images
        """
        for name in image_names:
            if name in self.image_paths:
                try:
                    self.loaded_images[name] = self.preprocessor.process_single_image(
                        self.image_paths[name]
                    )
                except Exception as e:
                    print(f"Error loading {name}: {e}")
            else:
                print(f"Image {name} not found in directory")
                
        return self.loaded_images
    
    def load_random_images(self, n: int = 5) -> Dict[str, np.ndarray]:
        """
        Load random selection of images
        
        Args:
            n: Number of random images to load
            
        Returns:
            Dictionary of loaded images
        """
        available_images = list(self.image_paths.keys())
        selected_images = random.sample(available_images, min(n, len(available_images)))
        return self.load_specific_images(selected_images)
    
    def compare_images(self, metric_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare loaded images using various metrics
        
        Args:
            metric_list: Optional list of specific metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        if not self.loaded_images:
            raise ValueError("No images loaded for comparison")
            
        default_metrics = ['histogram', 'noise', 'quality']
        metrics = metric_list or default_metrics
        results = []
        
        for name, image in self.loaded_images.items():
            metrics_dict = {'image_name': name}
            
            # Basic statistics
            stats = self.analyzer.compute_statistics(image)
            metrics_dict.update({
                'mean': np.mean(stats['mean']),
                'std': np.mean(stats['std']),
                'dynamic_range': np.mean(stats['dynamic_range'])
            })
            
            # Quality metrics
            quality = self.analyzer.assess_quality(image)
            metrics_dict.update(quality)
            
            # Noise analysis
            noise = self.analyzer.analyze_noise(image)
            metrics_dict.update(noise)
            
            results.append(metrics_dict)
            
        self.comparison_results = pd.DataFrame(results)
        return self.comparison_results
    
    def visualize_comparisons(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create visualization dashboard for image comparisons
        
        Args:
            figsize: Figure size for the dashboard
        """
        if self.comparison_results.empty:
            self.compare_images()
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Quality metrics comparison
        sns.barplot(
            data=self.comparison_results,
            x='image_name',
            y='blur_score',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Blur Score Comparison')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Noise level comparison
        sns.barplot(
            data=self.comparison_results,
            x='image_name',
            y='estimated_noise_level',
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Estimated Noise Level')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Dynamic range comparison
        sns.barplot(
            data=self.comparison_results,
            x='image_name',
            y='dynamic_range',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Dynamic Range')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Entropy comparison
        sns.barplot(
            data=self.comparison_results,
            x='image_name',
            y='entropy',
            ax=axes[1, 1]
        )
        axes[1, 1].set_title('Image Entropy')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def plot_image_grid(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Display loaded images in a grid
        
        Args:
            figsize: Figure size for the grid
        """
        n_images = len(self.loaded_images)
        if n_images == 0:
            raise ValueError("No images loaded to display")
            
        # Calculate grid dimensions
        n_cols = min(3, n_images)
        n_rows = (n_images - 1) // n_cols + 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.ravel()
        
        for idx, (name, image) in enumerate(self.loaded_images.items()):
            axes[idx].imshow(image)
            axes[idx].set_title(name)
            axes[idx].axis('off')
            
        # Hide empty subplots
        for idx in range(len(self.loaded_images), len(axes)):
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def generate_comparison_report(self) -> pd.DataFrame:
        """
        Generate detailed comparison report
        
        Returns:
            DataFrame with detailed statistics and comparisons
        """
        if self.comparison_results.empty:
            self.compare_images()
            
        # Add additional statistical measures
        report = self.comparison_results.copy()
        
        # Add rankings for each metric
        for column in report.select_dtypes(include=[np.number]).columns:
            report[f'{column}_rank'] = report[column].rank()
            
        # Add summary statistics
        summary_stats = report.describe()
        
        # Calculate z-scores for each metric
        for column in report.select_dtypes(include=[np.number]).columns:
            if not column.endswith('_rank'):
                report[f'{column}_zscore'] = stats.zscore(report[column])
                
        return report
    
    def export_results(self, output_dir: Union[str, Path]) -> None:
        """
        Export comparison results and visualizations
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export comparison results
        if not self.comparison_results.empty:
            self.comparison_results.to_csv(output_dir / 'comparison_results.csv')
            
        # Export detailed report
        report = self.generate_comparison_report()
        report.to_csv(output_dir / 'detailed_report.csv')
        
        # Save visualizations
        plt.figure(figsize=(15, 10))
        self.visualize_comparisons()
        plt.savefig(output_dir / 'comparison_visualization.png')
        
        plt.figure(figsize=(15, 10))
        self.plot_image_grid()
        plt.savefig(output_dir / 'image_grid.png')

class ImagePreprocessor:
    """Handles image preprocessing and standardization tasks."""
    
    def __init__(self, 
                 output_size: Tuple[int, int] = (1024, 1024),
                 resize_method: str = 'bicubic',
                 normalize: bool = True,
                 color_mode: str = 'rgb',
                 padding_color: Tuple[int, int, int] = (0, 0, 0)):
        """
        Initialize the image preprocessor.
        
        Args:
            output_size: Target size for processed images
            resize_method: Method for resizing ('nearest', 'bilinear', 'bicubic', 'lanczos')
            normalize: Whether to normalize pixel values to [0,1]
            color_mode: Target color mode ('rgb', 'bgr', 'grayscale')
            padding_color: Color to use for padding (RGB)
        """
        self.output_size = output_size
        self.resize_method = resize_method
        self.normalize = normalize
        self.color_mode = color_mode.lower()
        self.padding_color = padding_color
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Initialize parameters
        self._init_parameters()
        
    def _setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def _init_parameters(self):
        """Initialize internal parameters and validate settings."""
        # Supported image formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # Resize filter mapping
        self.filter_map = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'lanczos': Image.LANCZOS
        }
        
        # Validate and set resize method
        if self.resize_method.lower() not in self.filter_map:
            self.logger.warning(f"Invalid resize method '{self.resize_method}'. Using 'bicubic'.")
            self.resize_method = 'bicubic'
        
        self.resample_filter = self.filter_map[self.resize_method.lower()]
    
    def process_single_image(self, image_path: Union[str, Path], 
                           save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Process a single image file.
        
        Args:
            image_path: Path to input image
            save_path: Optional path to save processed image
            
        Returns:
            Processed image as numpy array
        """
        try:
            # Load image
            img = Image.open(image_path)
            
            # Convert color mode
            if self.color_mode == 'grayscale':
                img = img.convert('L')
            else:
                img = img.convert('RGB')
            
            # Resize while preserving aspect ratio
            img_resized = self._resize_preserve_aspect(img)
            
            # Convert to numpy array
            img_array = np.array(img_resized)
            
            # Normalize if requested
            if self.normalize:
                img_array = img_array.astype(np.float32) / 255.0
            
            # Handle color mode conversion
            if self.color_mode == 'bgr' and len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Save if path provided
            if save_path:
                self._save_processed_image(img_resized, save_path)
            
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            raise
    
    def process_directory(self, 
                         input_dir: Union[str, Path],
                         output_dir: Optional[Union[str, Path]] = None,
                         recursive: bool = False) -> Dict[str, np.ndarray]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Optional output directory path
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary of processed images {filename: image_array}
        """
        input_dir = Path(input_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect image files
        pattern = '**/*' if recursive else '*'
        image_files = [f for f in input_dir.glob(pattern) 
                      if f.suffix.lower() in self.supported_formats]
        
        if not image_files:
            self.logger.warning(f"No supported image files found in {input_dir}")
            return {}
        
        processed_images = {}
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                if output_dir:
                    save_path = output_dir / image_path.name
                else:
                    save_path = None
                
                processed = self.process_single_image(image_path, save_path)
                processed_images[image_path.name] = processed
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path.name}: {str(e)}")
                continue
        
        return processed_images
    
    def _resize_preserve_aspect(self, image: Image.Image) -> Image.Image:
        """Resize image preserving aspect ratio with padding."""
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        if aspect_ratio > 1:
            new_width = self.output_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = self.output_size[1]
            new_width = int(new_height * aspect_ratio)
        
        # Resize the image
        img_resized = image.resize((new_width, new_height), self.resample_filter)
        
        # Create background with padding color
        background = Image.new(image.mode, self.output_size, self.padding_color)
        
        # Calculate position to paste
        paste_x = (self.output_size[0] - new_width) // 2
        paste_y = (self.output_size[1] - new_height) // 2
        
        # Paste resized image onto background
        background.paste(img_resized, (paste_x, paste_y))
        
        return background
    
    def _save_processed_image(self, image: Image.Image, save_path: Union[str, Path]):
        """Save processed image with optimization."""
        save_path = Path(save_path)
        
        # Create directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with appropriate settings
        if save_path.suffix.lower() in {'.jpg', '.jpeg'}:
            image.save(save_path, quality=95, optimize=True)
        else:
            image.save(save_path, optimize=True)

class ImageAugmentor:
    """Utility class for image augmentation."""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        rows, cols = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, matrix, (cols, rows))
    
    @staticmethod
    def flip(image: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
        """Flip image horizontally or vertically."""
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        else:
            raise ValueError("direction must be 'horizontal' or 'vertical'")
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:,:,2] = hsv[:,:,2] * factor
        hsv[:,:,2] = np.clip(hsv[:,:,2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

class ImageAnalyzer:
    """Utility class for image analysis and quality assessment."""
    
    @staticmethod
    def compute_histogram(image: np.ndarray, channels: List[int] = None,
                         bins: int = 256, range_: tuple = (0, 256)) -> Dict[str, np.ndarray]:
        """
        Compute image histogram for specified channels.
        
        Args:
            image: Input image
            channels: List of channels to compute histograms for
            bins: Number of histogram bins
            range_: Range of values to include
            
        Returns:
            Dictionary of histograms for each channel
        """
        if channels is None:
            if len(image.shape) == 2:
                channels = [0]
            else:
                channels = list(range(image.shape[2]))
        
        histograms = {}
        for channel in channels:
            hist = cv2.calcHist([image], [channel], None, [bins], range_)
            histograms[f'channel_{channel}'] = hist.ravel()
            
        return histograms
    
    @staticmethod
    def compute_statistics(image: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute comprehensive image statistics.
        
        Returns dictionary containing:
        - Basic statistics (mean, std, min, max)
        - Dynamic range
        - Signal-to-noise ratio (SNR)
        - Root mean square (RMS)
        """
        stats = {
            'mean': np.mean(image, axis=(0,1)),
            'std': np.std(image, axis=(0,1)),
            'min': np.min(image, axis=(0,1)),
            'max': np.max(image, axis=(0,1)),
            'dynamic_range': np.ptp(image, axis=(0,1)),
            'rms': np.sqrt(np.mean(np.square(image), axis=(0,1)))
        }
        
        # Compute SNR if image is not empty
        if np.any(stats['std'] != 0):
            stats['snr'] = np.mean(stats['mean'] / stats['std'])
        else:
            stats['snr'] = float('inf')
            
        return stats
    
    @staticmethod
    def assess_quality(image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality using various metrics.
        
        Returns:
        - Blur detection (variance of Laplacian)
        - Contrast ratio
        - Entropy
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Compute blur metric (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_score = np.var(laplacian)
        
        # Compute contrast
        contrast = np.max(gray) - np.min(gray)
        
        # Compute entropy
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
        histogram = histogram.ravel() / histogram.sum()
        entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))
        
        return {
            'blur_score': blur_score,
            'contrast': contrast,
            'entropy': entropy
        }
    
    @staticmethod
    def detect_edges(image: np.ndarray, 
                    method: str = 'canny',
                    **kwargs) -> np.ndarray:
        """
        Detect edges in image using various methods.
        
        Supported methods:
        - 'canny': Canny edge detector
        - 'sobel': Sobel operator
        - 'laplacian': Laplacian operator
        
        Args:
            image: Input image
            method: Edge detection method
            **kwargs: Additional parameters for specific methods
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        if method == 'canny':
            low_threshold = kwargs.get('low_threshold', 100)
            high_threshold = kwargs.get('high_threshold', 200)
            return cv2.Canny(gray, low_threshold, high_threshold)
        
        elif method == 'sobel':
            ksize = kwargs.get('ksize', 3)
            scale = kwargs.get('scale', 1)
            delta = kwargs.get('delta', 0)
            
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=ksize, scale=scale, delta=delta)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=ksize, scale=scale, delta=delta)
            
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            
            return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        elif method == 'laplacian':
            ksize = kwargs.get('ksize', 3)
            laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
            return cv2.convertScaleAbs(laplacian)
        
        else:
            raise ValueError(f"Unsupported edge detection method: {method}")
    
    @staticmethod
    def compute_features(image: np.ndarray,
                        method: str = 'sift',
                        **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image features using various methods.
        
        Supported methods:
        - 'sift': Scale-Invariant Feature Transform
        - 'orb': Oriented FAST and Rotated BRIEF
        - 'brisk': Binary Robust Invariant Scalable Keypoints
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        if method == 'sift':
            detector = cv2.SIFT_create()
        elif method == 'orb':
            detector = cv2.ORB_create()
        elif method == 'brisk':
            detector = cv2.BRISK_create()
        else:
            raise ValueError(f"Unsupported feature detection method: {method}")
            
        keypoints, descriptors = detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    @staticmethod
    def analyze_noise(image: np.ndarray) -> Dict[str, float]:
        """
        Analyze image noise characteristics.
        
        Returns:
        - Noise level estimation
        - Signal-to-noise ratio
        - Noise distribution parameters
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Estimate noise using Laplacian operator
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_sigma = np.std(laplacian)
        
        # Compute local variance
        local_mean = cv2.blur(gray, (5,5))
        local_var = cv2.blur(np.square(gray - local_mean), (5,5))
        
        return {
            'estimated_noise_level': noise_sigma,
            'local_variance_mean': np.mean(local_var),
            'local_variance_std': np.std(local_var)
        }

# Example usage:
if __name__ == "__main__":
    # Initialize preprocessor with default settings
    preprocessor = ImagePreprocessor()
    
    # Process a directory of images
    processed_images = preprocessor.process_directory(
        input_dir="input_images",
        output_dir="processed_images"
    )
    
    # Initialize analyzer
    analyzer = ImageAnalyzer()
    
    # Analyze processed images
    for name, image in processed_images.items():
        stats = analyzer.compute_statistics(image)
        quality = analyzer.assess_quality(image)
        noise = analyzer.analyze_noise(image)
        
        print(f"Analysis for {name}:")
        print(f"Statistics: {stats}")
        print(f"Quality metrics: {quality}")
        print(f"Noise analysis: {noise}")