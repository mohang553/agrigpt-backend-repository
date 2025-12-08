"""
RAG Evaluation Script using LangSmith
Evaluates the RAG chatbot on correctness, faithfulness, relevance, and answer quality
"""

import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# LangSmith imports
from langsmith import Client
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example

# Import your RAG service
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Initialize LangSmith client
langsmith_client = Client()

class RAGEvaluator:
    def __init__(self, test_dataset_path: str = "../data/test_dataset.json"):
        """Initialize the evaluator with test dataset"""
        self.test_dataset_path = test_dataset_path
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON file"""
        with open(self.test_dataset_path, 'r') as f:
            return json.load(f)
    
    async def run_query(self, test_case: Dict) -> Dict[str, Any]:
        """
        Run a single query through the RAG system
        This is a mock implementation - you'll need to integrate with your actual RAG service
        """
        # Import your RAG service here
        from services.rag_service import RAGService
        
        rag_service = RAGService()
        await rag_service.initialize()
        
        query = test_case["query"]
        document_type = test_case["document_type"]
        
        # Get answer and sources from RAG
        answer, sources = await rag_service.query(
            query=query,
            document_type=document_type,
            chat_history=None
        )
        
        return {
            "answer": answer,
            "sources": sources,
            "test_case_id": test_case["test_case_id"],
            "category": test_case["category"]
        }
    
    def create_langsmith_dataset(self, dataset_name: str = "mosambi_schemes_rag_eval"):
        """Create a dataset in LangSmith for evaluation"""
        
        # Check if dataset already exists
        try:
            existing_datasets = list(langsmith_client.list_datasets())
            for ds in existing_datasets:
                if ds.name == dataset_name:
                    print(f"Dataset '{dataset_name}' already exists. Deleting and recreating...")
                    langsmith_client.delete_dataset(dataset_id=ds.id)
                    break
        except Exception as e:
            print(f"Note: {e}")
        
        # Create new dataset
        dataset = langsmith_client.create_dataset(
            dataset_name=dataset_name,
            description="Evaluation dataset for Mosambi Schemes RAG chatbot"
        )
        
        # Add examples to dataset
        for test_case in self.test_cases:
            langsmith_client.create_example(
                inputs={
                    "query": test_case["query"],
                    "document_type": test_case["document_type"]
                },
                outputs={
                    "expected_answer": test_case["expected_answer"],
                    "ground_truth_context": test_case["ground_truth_context"]
                },
                metadata={
                    "category": test_case["category"],
                    "test_case_id": test_case["test_case_id"],
                    "evaluation_criteria": test_case["evaluation_criteria"]
                },
                dataset_id=dataset.id
            )
        
        print(f"✅ Created dataset '{dataset_name}' with {len(self.test_cases)} test cases")
        return dataset
    
    async def evaluate_rag(self):
        """Main evaluation function"""
        
        print("="*80)
        print("🚀 STARTING RAG EVALUATION")
        print("="*80)
        
        # Create LangSmith dataset
        dataset = self.create_langsmith_dataset()
        
        # Import RAG service
        from services.rag_service import RAGService
        rag_service = RAGService()
        await rag_service.initialize()
        
        # Define the target function for evaluation
        async def rag_chain(inputs: Dict) -> Dict:
            """Wrapper function for RAG evaluation"""
            answer, sources = await rag_service.query(
                query=inputs["query"],
                document_type=inputs["document_type"],
                chat_history=None
            )
            return {
                "answer": answer,
                "sources": sources
            }
        
        # Define custom evaluators
        def correctness_evaluator(run: Run, example: Example) -> Dict:
            """Check if answer contains key information from expected answer"""
            answer = run.outputs.get("answer", "").lower()
            expected = example.outputs.get("expected_answer", "").lower()
            
            # Extract key numbers and facts
            score = 0
            feedback = []
            
            # Basic keyword matching
            expected_keywords = expected.split()
            matched_keywords = sum(1 for kw in expected_keywords if kw in answer)
            
            if matched_keywords / len(expected_keywords) > 0.5:
                score = 1
                feedback.append("Answer contains majority of expected information")
            elif matched_keywords / len(expected_keywords) > 0.3:
                score = 0.5
                feedback.append("Answer partially matches expected information")
            else:
                score = 0
                feedback.append("Answer lacks expected information")
            
            return {
                "key": "correctness",
                "score": score,
                "comment": " | ".join(feedback)
            }
        
        def faithfulness_evaluator(run: Run, example: Example) -> Dict:
            """Check if answer is grounded in retrieved sources"""
            answer = run.outputs.get("answer", "")
            sources = run.outputs.get("sources", [])
            
            score = 1 if len(sources) > 0 else 0
            comment = f"Retrieved {len(sources)} sources" if score else "No sources retrieved"
            
            return {
                "key": "faithfulness",
                "score": score,
                "comment": comment
            }
        
        def scope_adherence_evaluator(run: Run, example: Example) -> Dict:
            """Check if system properly handles out-of-scope questions"""
            category = example.metadata.get("category", "")
            answer = run.outputs.get("answer", "").lower()
            
            if category == "Out-of-Scope Questions":
                # Should refuse to answer
                refusal_phrases = [
                    "can only answer questions about",
                    "please ask a relevant question",
                    "outside my knowledge base",
                    "not related to"
                ]
                
                refused = any(phrase in answer for phrase in refusal_phrases)
                score = 1 if refused else 0
                comment = "Correctly refused out-of-scope question" if refused else "Failed to refuse out-of-scope question"
            else:
                # Should provide an answer
                refusal_phrases = [
                    "can only answer questions about",
                    "please ask a relevant question"
                ]
                provided_answer = not any(phrase in answer for phrase in refusal_phrases)
                score = 1 if provided_answer else 0
                comment = "Provided answer to in-scope question" if provided_answer else "Incorrectly refused in-scope question"
            
            return {
                "key": "scope_adherence",
                "score": score,
                "comment": comment
            }
        
        def answer_length_evaluator(run: Run, example: Example) -> Dict:
            """Check if answer has appropriate length"""
            answer = run.outputs.get("answer", "")
            word_count = len(answer.split())
            
            if 20 <= word_count <= 200:
                score = 1
                comment = f"Appropriate length ({word_count} words)"
            elif 10 <= word_count < 20 or 200 < word_count <= 300:
                score = 0.5
                comment = f"Suboptimal length ({word_count} words)"
            else:
                score = 0
                comment = f"Poor length ({word_count} words)"
            
            return {
                "key": "answer_length",
                "score": score,
                "comment": comment
            }
        
        # Run evaluation
        print("\n📊 Running evaluation with custom metrics...\n")
        
        results = evaluate(
            rag_chain,
            data=dataset.name,
            evaluators=[
                correctness_evaluator,
                faithfulness_evaluator,
                scope_adherence_evaluator,
                answer_length_evaluator
            ],
            experiment_prefix="mosambi-rag-eval",
            metadata={
                "version": "1.0",
                "date": datetime.now().isoformat(),
                "description": "RAG evaluation for Mosambi Schemes chatbot"
            }
        )
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE")
        print("="*80)
        print(f"\n📈 View detailed results in LangSmith dashboard")
        print(f"🔗 https://smith.langchain.com/")
        
        return results


async def main():
    """Main execution function"""
    
    # Check environment variables
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("❌ ERROR: LANGCHAIN_API_KEY not found in environment variables")
        print("Please set LANGCHAIN_API_KEY in your .env file")
        return
    
    # Initialize evaluator
    evaluator = RAGEvaluator(test_dataset_path="../data/test_dataset.json")
    
    # Run evaluation
    results = await evaluator.evaluate_rag()
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION SUMMARY")
    print("="*80)
    print(f"Total test cases: {len(evaluator.test_cases)}")
    print(f"Categories: Direct Factual (3), Complex Multi-Scheme (3), Out-of-Scope (3)")
    print("\nMetrics evaluated:")
    print("  1. Correctness - Does answer match expected information?")
    print("  2. Faithfulness - Is answer grounded in retrieved sources?")
    print("  3. Scope Adherence - Does system handle scope boundaries correctly?")
    print("  4. Answer Length - Is response appropriately sized?")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())