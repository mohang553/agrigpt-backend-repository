"""
Simple RAG Evaluation Script - Demo Version
This script runs evaluations on pre-collected results without needing full RAG service
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

class SimpleRAGEvaluator:
    def __init__(self, test_dataset_path: str = "../data/test_dataset.json"):
        """Initialize the evaluator with test dataset"""
        self.test_dataset_path = test_dataset_path
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON file"""
        with open(self.test_dataset_path, 'r') as f:
            return json.load(f)
    
    def evaluate_answer(self, test_case: Dict, actual_answer: str, sources: List[str]) -> Dict:
        """Evaluate a single answer against expected output"""
        
        category = test_case["category"]
        expected = test_case["expected_answer"].lower()
        actual = actual_answer.lower()
        criteria = test_case["evaluation_criteria"]
        
        results = {
            "test_case_id": test_case["test_case_id"],
            "category": category,
            "query": test_case["query"],
            "expected_answer": test_case["expected_answer"],
            "actual_answer": actual_answer,
            "sources": sources,
            "metrics": {}
        }
        
        # Metric 1: Correctness
        correctness_score = self._evaluate_correctness(actual, expected, criteria)
        results["metrics"]["correctness"] = correctness_score
        
        # Metric 2: Faithfulness (sources retrieved)
        faithfulness_score = self._evaluate_faithfulness(sources)
        results["metrics"]["faithfulness"] = faithfulness_score
        
        # Metric 3: Scope Adherence
        scope_score = self._evaluate_scope_adherence(actual, category)
        results["metrics"]["scope_adherence"] = scope_score
        
        # Metric 4: Answer Length
        length_score = self._evaluate_answer_length(actual_answer)
        results["metrics"]["answer_length"] = length_score
        
        # Calculate overall score
        results["overall_score"] = sum(results["metrics"].values()) / len(results["metrics"])
        
        return results
    
    def _evaluate_correctness(self, actual: str, expected: str, criteria: Dict) -> Dict:
        """Check if answer contains key information"""
        
        # Extract key phrases from expected answer
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        
        # Calculate word overlap
        overlap = len(expected_words & actual_words)
        total_expected = len(expected_words)
        
        if total_expected == 0:
            score = 0
        else:
            overlap_ratio = overlap / total_expected
            
            if overlap_ratio > 0.5:
                score = 1.0
                feedback = "✅ Answer contains majority of expected information"
            elif overlap_ratio > 0.3:
                score = 0.6
                feedback = "⚠️ Answer partially matches expected information"
            else:
                score = 0.2
                feedback = "❌ Answer lacks expected information"
        
        return {
            "score": score,
            "feedback": feedback,
            "overlap_ratio": overlap_ratio if total_expected > 0 else 0
        }
    
    def _evaluate_faithfulness(self, sources: List[str]) -> Dict:
        """Check if answer is grounded in retrieved sources"""
        
        if len(sources) >= 3:
            score = 1.0
            feedback = f"✅ Retrieved {len(sources)} sources (good grounding)"
        elif len(sources) >= 1:
            score = 0.7
            feedback = f"⚠️ Retrieved {len(sources)} sources (moderate grounding)"
        else:
            score = 0.0
            feedback = "❌ No sources retrieved (no grounding)"
        
        return {
            "score": score,
            "feedback": feedback,
            "sources_count": len(sources)
        }
    
    def _evaluate_scope_adherence(self, actual: str, category: str) -> Dict:
        """Check if system properly handles scope boundaries"""
        
        refusal_phrases = [
            "can only answer questions about",
            "please ask a relevant question",
            "outside my knowledge base",
            "not related to",
            "don't have enough information"
        ]
        
        contains_refusal = any(phrase in actual for phrase in refusal_phrases)
        
        if category == "Out-of-Scope Questions":
            # Should refuse
            if contains_refusal:
                score = 1.0
                feedback = "✅ Correctly refused out-of-scope question"
            else:
                score = 0.0
                feedback = "❌ Failed to refuse out-of-scope question"
        else:
            # Should provide answer
            if not contains_refusal:
                score = 1.0
                feedback = "✅ Provided answer to in-scope question"
            else:
                score = 0.0
                feedback = "❌ Incorrectly refused in-scope question"
        
        return {
            "score": score,
            "feedback": feedback,
            "is_refusal": contains_refusal
        }
    
    def _evaluate_answer_length(self, answer: str) -> Dict:
        """Check if answer has appropriate length"""
        
        word_count = len(answer.split())
        
        if 20 <= word_count <= 150:
            score = 1.0
            feedback = f"✅ Appropriate length ({word_count} words)"
        elif 10 <= word_count < 20 or 150 < word_count <= 250:
            score = 0.7
            feedback = f"⚠️ Suboptimal length ({word_count} words)"
        else:
            score = 0.3
            feedback = f"❌ Poor length ({word_count} words - too {'short' if word_count < 10 else 'long'})"
        
        return {
            "score": score,
            "feedback": feedback,
            "word_count": word_count
        }
    
    def run_evaluation_with_mock_data(self) -> List[Dict]:
        """
        Run evaluation with mock RAG responses
        In production, replace this with actual RAG service calls
        """
        
        # Mock responses for demonstration
        mock_responses = {
            "factual_1": {
                "answer": "The MIDH provides 60% subsidy in Year 1 for mosambi plantation, which amounts to Rs. 75,000 per hectare out of the total unit cost of Rs. 1,25,000.",
                "sources": ["Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 3)", "Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 4)"]
            },
            "factual_2": {
                "answer": "For horticultural crops under PMFBY, farmers pay a premium of 5% of the sum insured, while the government covers the remaining 95%.",
                "sources": ["Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 8)", "Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 9)"]
            },
            "factual_3": {
                "answer": "PM-KISAN provides Rs. 6,000 per year to eligible farmers, disbursed in three equal installments of Rs. 2,000 each.",
                "sources": ["Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 11)"]
            },
            "complex_1": {
                "answer": "SC/ST farmers in Telangana can maximize subsidies by combining multiple schemes: MIDH provides 60% subsidy (Rs. 1,50,000 for 2 hectares), PMKSY offers 100% drip irrigation subsidy (Rs. 1,20,000), PM-KISAN provides Rs. 6,000 annually, YSR Annadatha gives Rs. 18,000 per year, and NREGA convergence can provide approximately Rs. 30,000 for labor costs. The total support can reach around Rs. 3,24,000 in the first year.",
                "sources": ["Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 52)", "Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 13)", "Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 24)"]
            },
            "complex_2": {
                "answer": "MIDH area expansion subsidy eligibility requires: land ownership or a minimum 10-year registered lease deed, assured year-round irrigation source, no previous MIDH benefit in the last 5 years, and proper documentation. Priority is given to SC/ST farmers, women farmers, and small/marginal farmers.",
                "sources": ["Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 2)", "Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 3)"]
            },
            "complex_3": {
                "answer": "The total cost for establishing 1 hectare of mosambi plantation under MIDH is Rs. 1,25,000, which includes: plant material Rs. 16,500 (275 plants at Rs. 60 each), soil preparation Rs. 8,000, organic manures Rs. 20,000, inorganic fertilizers Rs. 25,000, bio-fertilizers Rs. 5,000, mulching materials Rs. 12,000, pesticides Rs. 18,000, labor charges Rs. 15,000, and basic micro irrigation Rs. 5,500.",
                "sources": ["Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 3)", "Comprehensive_Mosambi_Schemes_Guide.pdf (chunk 4)"]
            },
            "out_of_scope_1": {
                "answer": "I can only answer questions about Government agricultural schemes and programs. Please ask a relevant question about mosambi cultivation schemes or agricultural subsidies.",
                "sources": []
            },
            "out_of_scope_2": {
                "answer": "I can only answer questions about Government agricultural schemes and programs. Please ask a relevant question.",
                "sources": []
            },
            "out_of_scope_3": {
                "answer": "I can only answer questions about Government agricultural schemes and programs related to mosambi farming, not cooking or recipes. Please ask about schemes, subsidies, or agricultural support programs.",
                "sources": []
            }
        }
        
        results = []
        
        for test_case in self.test_cases:
            test_id = test_case["test_case_id"]
            mock_response = mock_responses.get(test_id, {
                "answer": "I don't have enough information in my knowledge base to answer that question.",
                "sources": []
            })
            
            # Evaluate the answer
            evaluation = self.evaluate_answer(
                test_case=test_case,
                actual_answer=mock_response["answer"],
                sources=mock_response["sources"]
            )
            
            results.append(evaluation)
        
        return results
    
    def print_results(self, results: List[Dict]):
        """Print evaluation results in a readable format"""
        
        print("\n" + "="*100)
        print("📊 RAG EVALUATION RESULTS - MOSAMBI SCHEMES CHATBOT")
        print("="*100)
        
        # Group by category
        categories = {}
        for result in results:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)
        
        # Print results by category
        for category, cat_results in categories.items():
            print(f"\n{'='*100}")
            print(f"📁 CATEGORY: {category}")
            print(f"{'='*100}")
            
            for i, result in enumerate(cat_results, 1):
                print(f"\n{'-'*100}")
                print(f"Test Case {i}/{len(cat_results)}: {result['test_case_id']}")
                print(f"{'-'*100}")
                print(f"\n❓ Query: {result['query']}")
                print(f"\n✅ Expected: {result['expected_answer'][:150]}...")
                print(f"\n💬 Actual: {result['actual_answer'][:150]}...")
                print(f"\n📚 Sources: {len(result['sources'])} sources retrieved")
                
                print(f"\n📊 METRICS:")
                print(f"  {'Metric':<20} {'Score':<10} {'Feedback'}")
                print(f"  {'-'*80}")
                
                for metric_name, metric_data in result['metrics'].items():
                    score_display = f"{metric_data['score']:.2f}"
                    print(f"  {metric_name:<20} {score_display:<10} {metric_data['feedback']}")
                
                print(f"\n  {'-'*80}")
                print(f"  {'OVERALL SCORE':<20} {result['overall_score']:.2f}/1.00")
        
        # Calculate aggregate scores
        print(f"\n\n{'='*100}")
        print("📈 AGGREGATE METRICS")
        print(f"{'='*100}")
        
        total_tests = len(results)
        avg_overall = sum(r['overall_score'] for r in results) / total_tests
        
        metric_names = list(results[0]['metrics'].keys())
        
        print(f"\n{'Metric':<25} {'Average Score':<15} {'Pass Rate (≥0.7)'}")
        print(f"{'-'*70}")
        
        for metric_name in metric_names:
            avg_score = sum(r['metrics'][metric_name]['score'] for r in results) / total_tests
            pass_count = sum(1 for r in results if r['metrics'][metric_name]['score'] >= 0.7)
            pass_rate = (pass_count / total_tests) * 100
            
            print(f"{metric_name:<25} {avg_score:.3f}           {pass_rate:.1f}% ({pass_count}/{total_tests})")
        
        print(f"{'-'*70}")
        print(f"{'OVERALL AVERAGE':<25} {avg_overall:.3f}")
        
        # Category-wise breakdown
        print(f"\n\n{'='*100}")
        print("📊 CATEGORY-WISE BREAKDOWN")
        print(f"{'='*100}")
        
        for category, cat_results in categories.items():
            avg_cat_score = sum(r['overall_score'] for r in cat_results) / len(cat_results)
            print(f"\n{category:<40} Avg Score: {avg_cat_score:.3f}  Tests: {len(cat_results)}")
        
        print(f"\n{'='*100}")
        print(f"✅ Evaluation completed on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}\n")
    
    def export_results(self, results: List[Dict], output_file: str = "evaluation_results.json"):
        """Export results to JSON file"""
        
        export_data = {
            "evaluation_date": datetime.now().isoformat(),
            "total_test_cases": len(results),
            "results": results,
            "aggregate_metrics": {
                "overall_average": sum(r['overall_score'] for r in results) / len(results),
                "by_metric": {}
            }
        }
        
        # Calculate aggregate by metric
        metric_names = list(results[0]['metrics'].keys())
        for metric_name in metric_names:
            avg_score = sum(r['metrics'][metric_name]['score'] for r in results) / len(results)
            export_data["aggregate_metrics"]["by_metric"][metric_name] = {
                "average_score": avg_score,
                "pass_rate": sum(1 for r in results if r['metrics'][metric_name]['score'] >= 0.7) / len(results)
            }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"📄 Results exported to: {output_file}")


def main():
    """Main execution function"""
    
    print("\n" + "="*100)
    print("🚀 STARTING RAG EVALUATION - DEMO MODE")
    print("="*100)
    print("\nThis demo uses pre-defined mock responses to demonstrate the evaluation framework.")
    print("In production, connect to your actual RAG service for live evaluation.\n")
    
    # Initialize evaluator
    evaluator = SimpleRAGEvaluator(test_dataset_path="../data/test_dataset.json")
    
    print(f"✅ Loaded {len(evaluator.test_cases)} test cases")
    print(f"📂 Categories: Direct Factual (3), Complex Multi-Scheme (3), Out-of-Scope (3)")
    
    # Run evaluation
    print("\n🔄 Running evaluation...")
    time.sleep(1)  # Simulate processing
    
    results = evaluator.run_evaluation_with_mock_data()
    
    # Print results
    evaluator.print_results(results)
    
    # Export results
    evaluator.export_results(results, "evaluation_results.json")
    
    print("\n✅ Evaluation complete!")
    print("\n💡 To run with your actual RAG service:")
    print("   1. Modify run_evaluation_with_mock_data() to call your RAG service")
    print("   2. Ensure your RAG service is initialized and running")
    print("   3. Update the mock_responses with actual API calls")


if __name__ == "__main__":
    main()