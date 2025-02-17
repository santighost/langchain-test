import time
import logging
import statistics
from datetime import datetime
from typing import Dict, List
import pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BedrockLoadTester:

    def __init__(self):
        load_dotenv()
        import importlib.util
        
        # Dynamically import the module
        spec = importlib.util.spec_from_file_location("nova_tests", "nova-tests.py")
        nova_tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(nova_tests)
        
        self.tester = nova_tests.BedrockTester()
        
        self.simple_prompt = "Hello there"
        self.system_prompt = "You are a helpful assistant that provides concise answers."
        self.test_duration = 600  # 10 minutes in seconds
        self.max_workers = 4     # Number of concurrent workers
        self.sleep_time = 1.0    # Delay between iterations
        self.results = []
    

    def single_test_iteration(self, test_type: str) -> Dict:
        """Execute a single test iteration"""
        start_time = time.time()
        
        try:
            if test_type == "direct":
                result = self.tester.direct_bedrock_call(
                    prompt=self.simple_prompt,
                    system=self.system_prompt
                )
            else:  # langchain
                result = self.tester.langchain_converse_call(
                    prompt=self.simple_prompt,
                    system=self.system_prompt
                )
                
            end_time = time.time()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "type": test_type,
                "total_latency": result["total_time"],
                "api_latency": result["converse_latency"],
                "init_latency": result["latency"],
                "success": True,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "timestamp": datetime.now().isoformat(),
                "type": test_type,
                "total_latency": end_time - start_time,
                "api_latency": None,
                "init_latency": None,
                "success": False,
                "error": str(e)
            }

    def run_load_test(self):
        """Run the load test for specified duration"""
        logger.info(f"Starting load test for {self.test_duration} seconds...")
        start_time = time.time()
        
        # Create a ThreadPoolExecutor for concurrent calls
        with ThreadPoolExecutor(max_workers=10) as executor:
            while time.time() - start_time < self.test_duration:
                # Submit both types of calls concurrently
                futures = []
                for test_type in ["direct", "langchain"]:
                    futures.append(
                        executor.submit(self.single_test_iteration, test_type)
                    )
                
                # Process results as they complete
                for future in futures:
                    result = future.result()
                    self.results.append(result)
                    
                    logger.info(
                        f"{result['type'].capitalize()} call - "
                        f"Total: {result['total_latency']:.2f}s, "
                        f"API: {result['api_latency']:.2f}s" if result['api_latency'] else "N/A"
                    )
                    
                # Small delay to avoid overwhelming local resources
                time.sleep(0.1)  # Reduced from 1s to 0.1s
    
    def single_test_iteration(self, test_type: str) -> Dict:
        """Execute a single test iteration"""
        start_time = time.time()
        
        try:
            if test_type == "direct":
                result = self.tester.direct_bedrock_call(
                    prompt=self.simple_prompt,
                    system=self.system_prompt
                )
            else:  # langchain
                result = self.tester.langchain_converse_call(
                    prompt=self.simple_prompt,
                    system=self.system_prompt
                )
                
            end_time = time.time()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "type": test_type,
                "total_latency": result["total_time"],
                "api_latency": result["converse_latency"],
                "init_latency": result["latency"],
                "success": True,
                "error": None
            }
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error in {test_type} call: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "type": test_type,
                "total_latency": end_time - start_time,
                "api_latency": None,
                "init_latency": None,
                "success": False,
                "error": str(e)
            }
    

    def analyze_results(self):
        """Analyze and visualize test results"""
        df = pd.DataFrame(self.results)
        
        # Split results by type and remove first occurrence of each type
        direct_results = df[df['type'] == 'direct'].iloc[1:]  # Skip first direct call
        langchain_results = df[df['type'] == 'langchain'].iloc[1:]  # Skip first langchain call
        
        # Calculate statistics (excluding first calls)
        stats = {
            'direct': {
                'total_latency': {
                    'mean': direct_results['total_latency'].mean(),
                    'median': direct_results['total_latency'].median(),
                    'std': direct_results['total_latency'].std(),
                    'success_rate': (direct_results['success'].sum() / len(direct_results)) * 100
                }
            },
            'langchain': {
                'total_latency': {
                    'mean': langchain_results['total_latency'].mean(),
                    'median': langchain_results['total_latency'].median(),
                    'std': langchain_results['total_latency'].std(),
                    'success_rate': (langchain_results['success'].sum() / len(langchain_results)) * 100
                }
            }
        }
        
        # Print statistics
        logger.info("\nTest Results Summary (excluding first calls):")
        for test_type, metrics in stats.items():
            logger.info(f"\n{test_type.capitalize()} Results:")
            logger.info(f"Mean latency: {metrics['total_latency']['mean']:.2f}s")
            logger.info(f"Median latency: {metrics['total_latency']['median']:.2f}s")
            logger.info(f"Std deviation: {metrics['total_latency']['std']:.2f}s")
            logger.info(f"Success rate: {metrics['total_latency']['success_rate']:.1f}%")
        
        # Create visualization with call sequence instead of timestamps
        plt.figure(figsize=(12, 6))
        
        # Use call sequence numbers instead of timestamps
        direct_x = range(len(direct_results))
        langchain_x = range(len(langchain_results))
        
        plt.plot(direct_x, direct_results['total_latency'], 
                label='Direct', alpha=0.6)
        plt.plot(langchain_x, langchain_results['total_latency'], 
                label='LangChain', alpha=0.6)
        plt.xlabel('Call Sequence')
        plt.ylabel('Latency (seconds)')
        plt.title('Latency Over Time (excluding initialization calls)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('latency_comparison.png')

        
        # Save results to CSV (including all data for reference)
        df.to_csv('load_test_results.csv', index=False)
        
        # Also save filtered results (excluding first calls)
        filtered_df = pd.concat([direct_results, langchain_results])
        filtered_df.to_csv('load_test_results_filtered.csv', index=False)
        
        logger.info("\nResults saved to:")
        logger.info("- 'load_test_results.csv' (all data)")
        logger.info("- 'load_test_results_filtered.csv' (excluding first calls)")
        logger.info("- 'latency_comparison.png'")

def main():
    load_tester = BedrockLoadTester()
    load_tester.run_load_test()
    load_tester.analyze_results()

if __name__ == "__main__":
    main()