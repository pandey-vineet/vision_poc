import cv2
import numpy as np
from paddleocr import PaddleOCR
import os


class PaddleOCRProcessor:
    """
    A class for processing images with PaddleOCR, including preprocessing,
    OCR detection, visualization, and text extraction.
    """
    
    def __init__(self, use_textline_orientation=True, lang='en'):
        """
        Initialize PaddleOCR processor with optimized settings.
        
        Args:
            use_textline_orientation: Use textline orientation (helps with rotated text)
            lang: Language code ('en', 'ch', 'korean', etc.)
        """
        self.ocr = PaddleOCR(
            use_textline_orientation=use_textline_orientation,
            lang=lang,
        )
        self.use_textline_orientation = use_textline_orientation
        self.lang = lang
    
    def preprocess_image(self, image_path, method='adaptive'):
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image_path: Path to input image
            method: Preprocessing method ('adaptive', 'otsu', 'enhanced')
        
        Returns:
            Preprocessed image
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Upscale for better small text recognition (optional, can be tuned)
        scale_factor = 2.0
        if scale_factor > 1.0:
            gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, 
                             interpolation=cv2.INTER_CUBIC)
        
        if method == 'adaptive':
            # Method 1: Adaptive thresholding (good for varying lighting)
            # Apply slight denoising first
            denoised = cv2.fastNlMeansDenoising(gray, None, h=10, 
                                                templateWindowSize=7, 
                                                searchWindowSize=21)
            
            # Adaptive threshold
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
        elif method == 'otsu':
            # Method 2: Otsu's thresholding (good for clear text)
            # Denoise
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
        elif method == 'enhanced':
            # Method 3: Enhanced preprocessing (best for difficult images)
            # 1. Illumination correction
            bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
            norm = cv2.divide(gray, bg, scale=255)
            norm = np.clip(norm, 0, 255).astype(np.uint8)
            
            # 2. Local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast = clahe.apply(norm)
            
            # 3. Denoise
            denoised = cv2.fastNlMeansDenoising(contrast, None, h=12, 
                                                templateWindowSize=7, 
                                                searchWindowSize=21)
            
            # 4. Adaptive threshold
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 35, 8
            )
            
            # 5. Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        else:
            binary = gray
        
        return binary
    
    def run_ocr_with_preprocessing(self, image_path, preprocessing_method='adaptive', 
                                   min_confidence=0.5):
        """
        Run OCR on preprocessed image.
        
        Args:
            image_path: Path to input image
            preprocessing_method: Preprocessing method to use
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            List of detected text with confidence scores
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image_path, method=preprocessing_method)
        
        # Save preprocessed image for debugging
        output_path = f"preprocessed_{os.path.basename(image_path)}"
        cv2.imwrite(output_path, preprocessed)
        print(f"Preprocessed image saved to: {output_path}")
        
        # Convert grayscale to BGR (3-channel) if needed
        # PaddleOCR expects 3-channel images
        if len(preprocessed.shape) == 2:
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        
        # Run OCR on preprocessed image using predict() method
        # Pass file path instead of numpy array to avoid channel issues
        result = self.ocr.predict(input=output_path)
        
        # Parse and filter results
        # predict() returns a list of result dict-like objects
        detected_texts = []
        if result:
            for res in result:
                # Each result has 'rec_texts' (list of texts) and 'rec_scores' (list of scores)
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', [])
                rec_polys = res.get('rec_polys', [])
                
                # Match texts with scores and boxes
                for i, text in enumerate(rec_texts):
                    if i < len(rec_scores):
                        confidence = rec_scores[i]
                        # Confidence is already 0-1, so compare directly
                        if confidence >= min_confidence:
                            bbox = rec_polys[i] if i < len(rec_polys) else None
                            detected_texts.append({
                                'text': text,
                                'confidence': confidence,
                                'bbox': bbox
                            })
        
        return detected_texts
    
    def run_ocr_original(self, image_path, min_confidence=0.5):
        """
        Run OCR on original image (no preprocessing).
        
        Args:
            image_path: Path to input image
            min_confidence: Minimum confidence threshold (0-1)
        
        Returns:
            List of detected text with confidence scores
        """
        # Use predict() method instead of deprecated ocr() method
        result = self.ocr.predict(input=image_path)
        
        # Parse and filter results
        # predict() returns a list of result dict-like objects
        detected_texts = []
        if result:
            for res in result:
                # Each result has 'rec_texts' (list of texts) and 'rec_scores' (list of scores)
                rec_texts = res.get('rec_texts', [])
                rec_scores = res.get('rec_scores', [])
                rec_polys = res.get('rec_polys', [])
                
                # Match texts with scores and boxes
                for i, text in enumerate(rec_texts):
                    if i < len(rec_scores):
                        confidence = rec_scores[i]
                        # Confidence is already 0-1, so compare directly
                        if confidence >= min_confidence:
                            bbox = rec_polys[i] if i < len(rec_polys) else None
                            detected_texts.append({
                                'text': text,
                                'confidence': confidence,
                                'bbox': bbox
                            })
        
        return detected_texts
    
    def visualize_detections(self, image_path, detected_texts, output_path=None):
        """
        Draw bounding boxes and text labels on the image and save it.
        
        Args:
            image_path: Path to the original image
            detected_texts: List of detected text dictionaries with 'text', 'confidence', and 'bbox'
            output_path: Path to save the annotated image (default: adds '_annotated' to input name)
        
        Returns:
            Path to the saved annotated image
        """
        # Read the original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Create a copy for drawing
        annotated_img = img.copy()
        
        # Draw each detection
        for i, detection in enumerate(detected_texts):
            text = detection['text']
            confidence = detection['confidence']
            bbox = detection.get('bbox')
            
            if bbox is not None:
                try:
                    # Convert bbox to numpy array if it's a list
                    if isinstance(bbox, list):
                        bbox = np.array(bbox, dtype=np.int32)
                    elif not isinstance(bbox, np.ndarray):
                        bbox = np.array(bbox, dtype=np.int32)
                    
                    # Ensure it's int32
                    if bbox.dtype != np.int32:
                        bbox = bbox.astype(np.int32)
                    
                    # Handle different bbox formats
                    if len(bbox.shape) == 2 and bbox.shape[0] >= 4:
                        # Polygon format (4 or more points) - shape (n, 2)
                        # Reshape to ensure correct format: (1, n, 2)
                        bbox_reshaped = bbox.reshape(1, -1, 2)
                        # Draw the polygon
                        cv2.polylines(annotated_img, bbox_reshaped, isClosed=True, 
                                    color=(0, 255, 0), thickness=2)
                        
                        # Get bounding rectangle for text placement
                        x, y, w, h = cv2.boundingRect(bbox)
                    elif len(bbox.shape) == 1 and len(bbox) == 4:
                        # Rectangle format [x, y, w, h] or [x1, y1, x2, y2]
                        if isinstance(bbox[0], (list, tuple, np.ndarray)):
                            # It's a list of points, get bounding rect
                            bbox_array = np.array(bbox, dtype=np.int32)
                            bbox_reshaped = bbox_array.reshape(1, -1, 2)
                            x, y, w, h = cv2.boundingRect(bbox_array)
                            cv2.polylines(annotated_img, bbox_reshaped, isClosed=True,
                                        color=(0, 255, 0), thickness=2)
                        else:
                            # Assume [x, y, w, h] format
                            x, y, w, h = map(int, bbox)
                            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), 
                                        color=(0, 255, 0), thickness=2)
                    else:
                        # Fallback: use bounding rect if available
                        if len(bbox.shape) == 2:
                            x, y, w, h = cv2.boundingRect(bbox)
                            bbox_reshaped = bbox.reshape(1, -1, 2)
                            cv2.polylines(annotated_img, bbox_reshaped, isClosed=True,
                                        color=(0, 255, 0), thickness=2)
                        else:
                            # Skip if bbox format is not recognized
                            x, y, w, h = 10, 10 + i * 30, 200, 25
                except Exception as e:
                    # If bbox processing fails, use default position
                    print(f"Warning: Could not process bbox for '{text}': {e}")
                    x, y, w, h = 10, 10 + i * 30, 200, 25
                
                # Prepare text label with confidence
                label = f"{text} ({confidence:.2f})"
                
                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw text background
                cv2.rectangle(annotated_img,
                             (x, y - text_height - baseline - 5),
                             (x + text_width, y),
                             color=(0, 255, 0),
                             thickness=-1)
                
                # Draw text
                cv2.putText(annotated_img, label,
                           (x, y - baseline - 5),
                           font, font_scale,
                           color=(0, 0, 0),
                           thickness=thickness)
            else:
                # If no bbox, just print text at a default location
                label = f"{text} ({confidence:.2f})"
                cv2.putText(annotated_img, label,
                           (10, 30 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                           color=(0, 255, 0), thickness=2)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_annotated.png"
        
        # Save the annotated image
        cv2.imwrite(output_path, annotated_img)
        print(f"\nAnnotated image saved to: {output_path}")
        
        return output_path
    
    def print_high_confidence_texts(self, detected_texts, confidence_threshold=0.7):
        """
        Print texts that have confidence higher than the specified threshold.
        
        Args:
            detected_texts: List of detected text dictionaries with 'text' and 'confidence'
            confidence_threshold: Minimum confidence threshold (0-1) to filter texts
        
        Returns:
            List of high-confidence texts
        """
        # Filter texts above threshold
        high_confidence_texts = [
            item for item in detected_texts 
            if item['confidence'] >= confidence_threshold
        ]
        
        # Sort by confidence (highest first)
        high_confidence_texts.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\n{'=' * 60}")
        print(f"Texts with confidence >= {confidence_threshold:.2f}:")
        print(f"{'=' * 60}")
        
        if high_confidence_texts:
            print(f"Found {len(high_confidence_texts)} high-confidence detections:\n")
            for i, item in enumerate(high_confidence_texts, 1):
                print(f"  {i}. {item['text']:20s} | Confidence: {item['confidence']:.3f}")
        else:
            print(f"No texts found with confidence >= {confidence_threshold:.2f}")
        
        print(f"{'=' * 60}\n")
        
        return high_confidence_texts
    
    def split_texts_by_spaces(self, detected_texts):
        """
        Create a list of detected texts where texts with spaces are split into separate entries.
        
        Args:
            detected_texts: List of detected text dictionaries with 'text', 'confidence', and 'bbox'
        
        Returns:
            List of detected texts with spaces split into separate entries
        """
        split_texts = []
        
        for item in detected_texts:
            text = item['text']
            confidence = item['confidence']
            bbox = item.get('bbox')
            
            # Check if text contains spaces
            if ' ' in text:
                # Split by spaces
                words = text.split()
                for word in words:
                    # Create a new entry for each word
                    split_texts.append({
                        'text': word,
                        'confidence': confidence,
                        'bbox': bbox  # Keep the same bbox (or could be None)
                    })
            else:
                # No spaces, keep as is
                split_texts.append({
                    'text': text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        return split_texts
    
    def process_image(self, image_path, preprocessing_methods=None, min_confidence=0.5,
                     confidence_threshold=0.95, save_results=True):
        """
        Process an image with multiple preprocessing methods and return the best results.
        
        Args:
            image_path: Path to input image
            preprocessing_methods: List of preprocessing methods to try (default: ['original', 'adaptive', 'enhanced'])
            min_confidence: Minimum confidence threshold for OCR (0-1)
            confidence_threshold: Minimum confidence threshold for filtering high-confidence texts (0-1)
            save_results: Whether to save text lists and annotated images
        
        Returns:
            Dictionary containing:
                - 'best_results': List of all detected texts from best method
                - 'high_confidence_texts': List of high-confidence texts
                - 'split_texts': List of high-confidence texts split by spaces
                - 'text_list': List of text strings (split by spaces)
                - 'original_text_list': List of text strings (not split)
                - 'annotated_path': Path to annotated image
        """
        if preprocessing_methods is None:
            preprocessing_methods = ['original', 'adaptive', 'enhanced']
        
        print(f"\nProcessing image: {image_path}")
        print("=" * 60)
        
        all_results = {}
        
        # Try original image
        if 'original' in preprocessing_methods:
            print("\n1. OCR on original image:")
            results_original = self.run_ocr_original(image_path, min_confidence=min_confidence)
            all_results['original'] = results_original
            for item in results_original:
                print(f"   Text: {item['text']} | Confidence: {item['confidence']:.3f}")
        
        # Try adaptive preprocessing
        if 'adaptive' in preprocessing_methods:
            print("\n2. OCR on preprocessed image (adaptive thresholding):")
            results_preprocessed = self.run_ocr_with_preprocessing(
                image_path, preprocessing_method='adaptive', min_confidence=min_confidence
            )
            all_results['adaptive'] = results_preprocessed
            for item in results_preprocessed:
                print(f"   Text: {item['text']} | Confidence: {item['confidence']:.3f}")
        
        # Try enhanced preprocessing
        if 'enhanced' in preprocessing_methods:
            print("\n3. OCR on enhanced preprocessed image:")
            results_enhanced = self.run_ocr_with_preprocessing(
                image_path, preprocessing_method='enhanced', min_confidence=min_confidence
            )
            all_results['enhanced'] = results_enhanced
            for item in results_enhanced:
                print(f"   Text: {item['text']} | Confidence: {item['confidence']:.3f}")
        
        # Compare results
        print("\n" + "=" * 60)
        print("Summary:")
        for method, results in all_results.items():
            print(f"  {method.capitalize()}: {len(results)} detections")
        
        # Get best results (method with most detections)
        best_method = max(all_results.keys(), key=lambda k: len(all_results[k]))
        best_results = all_results[best_method]
        
        if best_results:
            print(f"\nBest method ({best_method}) found {len(best_results)} detections")
            print("\nAll detected text:")
            for item in best_results:
                print(f"  - {item['text']} (confidence: {item['confidence']:.3f})")
            
            # Get high-confidence texts
            high_confidence_texts = self.print_high_confidence_texts(
                best_results, confidence_threshold=confidence_threshold
            )
            
            # Split high-confidence texts by spaces
            split_high_confidence_texts = self.split_texts_by_spaces(high_confidence_texts)
            
            # Create list of split texts (by spaces)
            text_list = []
            if split_high_confidence_texts:
                for item in split_high_confidence_texts:
                    text_list.append(item['text'])
            print(f"collected txt items from the image (split by spaces): {text_list}")
            
            # Create list of original texts (not split by spaces)
            original_text_list = []
            if high_confidence_texts:
                for item in high_confidence_texts:
                    original_text_list.append(item['text'])
            print(f"collected txt items from the image (not split): {original_text_list}")
            
            # Save results if requested
            annotated_path = None
            if save_results:
                # Save split text_list to a .txt file
                output_txt_file = "test_list.txt"
                with open(output_txt_file, 'w', encoding='utf-8') as f:
                    for text in text_list:
                        f.write(text + '\n')
                print(f"Text list (split by spaces) saved to: {output_txt_file}")
                
                # Save original text_list to a separate .txt file
                output_txt_file_original = "test_list_original.txt"
                with open(output_txt_file_original, 'w', encoding='utf-8') as f:
                    for text in original_text_list:
                        f.write(text + '\n')
                print(f"Text list (not split by spaces) saved to: {output_txt_file_original}")
                
                # Save annotated image
                print("\n" + "=" * 60)
                annotated_path = self.visualize_detections(image_path, best_results)
                print(f"Visualization saved to: {annotated_path}")
            
            return {
                'best_results': best_results,
                'high_confidence_texts': high_confidence_texts,
                'split_texts': split_high_confidence_texts,
                'text_list': text_list,
                'original_text_list': original_text_list,
                'annotated_path': annotated_path,
                'best_method': best_method
            }
        else:
            print("\nNo detections found with any method.")
            return {
                'best_results': [],
                'high_confidence_texts': [],
                'split_texts': [],
                'text_list': [],
                'original_text_list': [],
                'annotated_path': None,
                'best_method': None
            }


if __name__ == "__main__":
    # Example usage
    print("Initializing PaddleOCR...")
    processor = PaddleOCRProcessor(
        use_textline_orientation=True,
        lang='en',
    )
    
    # Process an image
    image_path = "t_1.png"
    results = processor.process_image(
        image_path,
        preprocessing_methods=['original', 'adaptive', 'enhanced'],
        min_confidence=0.5,
        confidence_threshold=0.95,
        save_results=True
    )
