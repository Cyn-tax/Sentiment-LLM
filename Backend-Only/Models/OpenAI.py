"""
OpenAI Sentiment Analysis Model
Processes customer feedback from CSV and classifies sentiment using OpenAI API.
"""

import os
import csv
import json
from typing import Dict, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()


class SentimentAnalyzer:
    """Sentiment analysis model using OpenAI API."""
    
    def __init__(self, model: str = "gpt-5"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model: OpenAI model to use (default: gpt-5)
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.results: List[Dict] = []
    
    def analyze_sentiment(self, feedback_text: str) -> str:
        """
        Analyze sentiment of a single feedback text.
        
        Args:
            feedback_text: The customer feedback text to analyze
            
        Returns:
            Sentiment classification: "Positive", "Negative", or "Neutral"
        """
        system_prompt = """You are a sentiment analysis expert. Analyze customer feedback and classify the sentiment.

You must respond with a JSON object containing only the "sentiment" field with one of these exact values: "Positive", "Negative", or "Neutral".

Guidelines:
- Positive: Expresses satisfaction, praise, appreciation, or positive emotions
- Negative: Expresses dissatisfaction, criticism, complaints, or negative emotions
- Neutral: Factual statements, questions, requests for information, or neutral content without clear emotional tone"""

        user_prompt = f"""Analyze the following customer feedback and classify its sentiment.

Customer Feedback: "{feedback_text}"

Respond with a JSON object in this exact format:
{{"sentiment": "Positive"}}
or
{{"sentiment": "Negative"}}
or
{{"sentiment": "Neutral"}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the JSON response
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Extract sentiment
            sentiment = result.get("sentiment", "Neutral").strip()
            
            # Validate and normalize
            sentiment = sentiment.capitalize()
            if sentiment not in ["Positive", "Negative", "Neutral"]:
                # Fallback parsing
                content_upper = content.upper()
                if "POSITIVE" in content_upper:
                    sentiment = "Positive"
                elif "NEGATIVE" in content_upper:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
            
            return sentiment
            
        except json.JSONDecodeError as e:
            content = response.choices[0].message.content if 'response' in locals() else "N/A"
            print(f"JSON decode error: {e}")
            print(f"Response content: {content}")
            return "Neutral"
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return "Neutral"
    
    def process_csv(self, csv_path: str, start_row: int = 1, end_row: Optional[int] = None, progress_callback=None):
        """
        Process CSV file row by row and analyze sentiment.
        
        Args:
            csv_path: Path to the CSV file
            start_row: Starting row index (1-indexed, excluding header)
            end_row: Ending row index (1-indexed, excluding header). None means process all rows.
            progress_callback: Optional callback function(current, total, row_num) called after each row
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        self.results = []
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Convert to list to enable indexing
            rows = list(reader)
            
            # Determine range
            if end_row is None:
                end_row = len(rows)
            else:
                end_row = min(end_row, len(rows))
            
            total_rows = end_row - (start_row - 1)
            processed_count = 0
            
            # Process rows one at a time
            for idx in range(start_row - 1, end_row):
                if idx >= len(rows):
                    break
                
                row = rows[idx]
                feedback_text = row.get('feedback_text', '').strip()
                
                if not feedback_text:
                    continue
                
                print(f"Processing row {idx + 2}...")  # +2 because idx is 0-based and we skip header
                
                # Analyze sentiment
                predicted_sentiment = self.analyze_sentiment(feedback_text)
                
                # Store result
                result = {
                    "row": idx + 2,  # 1-indexed row number (including header)
                    "feedback_text": feedback_text,
                    "predicted_sentiment": predicted_sentiment,
                    "true_sentiment": row.get('true_sentiment', 'N/A')
                }
                
                self.results.append(result)
                processed_count += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(processed_count, total_rows, idx + 2)
                
                print(f"  Feedback: {feedback_text[:50]}...")
                print(f"  Predicted: {predicted_sentiment}")
                print()
    
    def calculate_confusion_matrix(self) -> Dict:
        """
        Calculate confusion matrix comparing true vs predicted sentiments.
        
        Returns:
            Dictionary containing confusion matrix data
        """
        # Initialize confusion matrix
        sentiments = ["Positive", "Negative", "Neutral"]
        confusion_matrix = {true_sent: {pred_sent: 0 for pred_sent in sentiments} 
                           for true_sent in sentiments}
        
        # Count occurrences
        for result in self.results:
            true_sentiment = result.get('true_sentiment', 'N/A').strip()
            predicted_sentiment = result.get('predicted_sentiment', 'N/A').strip()
            
            # Normalize sentiments
            true_sentiment = true_sentiment.capitalize() if true_sentiment != 'N/A' else 'N/A'
            predicted_sentiment = predicted_sentiment.capitalize() if predicted_sentiment != 'N/A' else 'N/A'
            
            # Only count if both are valid sentiments
            if true_sentiment in sentiments and predicted_sentiment in sentiments:
                confusion_matrix[true_sentiment][predicted_sentiment] += 1
        
        # Calculate metrics
        total_correct = sum(confusion_matrix[sent][sent] for sent in sentiments)
        total_processed = sum(sum(confusion_matrix[sent].values()) for sent in sentiments)
        accuracy = (total_correct / total_processed * 100) if total_processed > 0 else 0
        
        # Calculate per-class metrics
        class_metrics = {}
        for sentiment in sentiments:
            true_positives = confusion_matrix[sentiment][sentiment]
            false_positives = sum(confusion_matrix[other][sentiment] for other in sentiments if other != sentiment)
            false_negatives = sum(confusion_matrix[sentiment][other] for other in sentiments if other != sentiment)
            true_negatives = total_processed - true_positives - false_positives - false_negatives
            
            precision = (true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0
            recall = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0
            f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            class_metrics[sentiment] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "true_negatives": true_negatives
            }
        
        return {
            "matrix": confusion_matrix,
            "accuracy": round(accuracy, 4),
            "total_correct": total_correct,
            "total_processed": total_processed,
            "class_metrics": class_metrics
        }
    
    def save_results(self, output_path: str = "results.json"):
        """
        Save analysis results to JSON file.
        
        Args:
            output_path: Path to save the results JSON file
        """
        output_path = Path(output_path)
        
        # Calculate confusion matrix
        confusion_matrix_data = self.calculate_confusion_matrix()
        
        output_data = {
            "total_processed": len(self.results),
            "results": self.results,
            "confusion_matrix": confusion_matrix_data
        }
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
        return output_path


def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = SentimentAnalyzer(model="gpt-5")
    
    # Process CSV (all rows)
    csv_path = Path(__file__).parent.parent.parent / "Data.csv"
    analyzer.process_csv(csv_path, start_row=1, end_row=None)
    
    # Save results
    results_path = Path(__file__).parent.parent.parent / "results.json"
    analyzer.save_results(results_path)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Total rows processed: {len(analyzer.results)}")
    
    sentiment_counts = {}
    for result in analyzer.results:
        sentiment = result['predicted_sentiment']
        sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    print("\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count}")
    
    # Print confusion matrix summary
    confusion_matrix_data = analyzer.calculate_confusion_matrix()
    print("\n" + "="*50)
    print("CONFUSION MATRIX")
    print("="*50)
    print(f"Overall Accuracy: {confusion_matrix_data['accuracy']}%")
    print(f"Total Correct: {confusion_matrix_data['total_correct']}/{confusion_matrix_data['total_processed']}")
    
    print("\nConfusion Matrix (Actual vs Predicted):")
    print(f"{'Actual\\Predicted':<15} {'Positive':<10} {'Negative':<10} {'Neutral':<10}")
    print("-" * 50)
    for true_sent in ["Positive", "Negative", "Neutral"]:
        row = f"{true_sent:<15}"
        for pred_sent in ["Positive", "Negative", "Neutral"]:
            count = confusion_matrix_data['matrix'][true_sent][pred_sent]
            row += f"{count:<10}"
        print(row)
    
    print("\nPer-Class Metrics:")
    for sentiment, metrics in confusion_matrix_data['class_metrics'].items():
        print(f"\n{sentiment}:")
        print(f"  Precision: {metrics['precision']}")
        print(f"  Recall: {metrics['recall']}")
        print(f"  F1-Score: {metrics['f1_score']}")


if __name__ == "__main__":
    main()

