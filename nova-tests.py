import os
import time
import logging
import statistics
import warnings
import boto3
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Environment variables setup remains the same
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_SESSION_TOKEN'] = ''
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Get the root logger and set its level to INFO
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create main logger for INFO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Langchain logger
logging.getLogger('langchain_aws').setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

class BedrockTester:
    def __init__(self):
        self.model_id = "us.amazon.nova-lite-v1:0"
        self.bedrock_client = self._create_bedrock_client()
        
        self.test_cases = [
            {
                "name": "Simple Question",
                "prompt": "What is the capital of France?",
                "system": "You are a helpful assistant that provides concise answers.",
            },
            {
                "name": "Technical Explanation",
                "prompt": "Explain quantum computing in one sentence.",
                "system": "You are a technical expert providing precise explanations.",
            },
            {
                "name": "Creative Task",
                "prompt": "Write a haiku about programming.",
                "system": "You are a creative writer specializing in short-form poetry.",
            }
        ]

    def _create_bedrock_client(self) -> Any:
        try:
            return boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise

    # Call Bedrock directly using Converse API
    def direct_bedrock_call(self, prompt: str, system: str) -> Dict:
        try:
            total_start = time.time()
            initial_client_id = id(self.bedrock_client)
            
            # Track initialization time
            init_start = time.time()
            messages = [
                {
                    "role": "user",
                    "content": [{"text": prompt}]
                }
            ]
            inf_params = {
                "maxTokens": 512,
                "topP": 0.9,
                "temperature": 0.7
            }
            init_time = time.time() - init_start
            
            # Track API call time
            pre_call_client_id = id(self.bedrock_client)
            api_start = time.time()
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=messages,
                system=[{"text": system}],
                inferenceConfig=inf_params
            )
            api_time = time.time() - api_start
            post_call_client_id = id(self.bedrock_client)
            
            total_time = time.time() - total_start
            
            logger.debug(f"Direct Bedrock timing breakdown:")
            logger.debug(f"  Initialization: {init_time:.3f}s")
            logger.debug(f"  API call: {api_time:.3f}s")
            logger.debug(f"  Total time: {total_time:.3f}s")
            
            return {
                "response": response["output"]["message"]["content"][0]["text"],
                "latency": init_time,
                "converse_latency": api_time,
                "total_time": total_time,
                "client_reused": initial_client_id == pre_call_client_id,
                "converse_client_reused": pre_call_client_id == post_call_client_id,
                "timing_breakdown": {
                    "init": init_time,
                    "api": api_time,
                    "total": total_time
                }
            }
        except Exception as e:
            logger.error(f"Unexpected error in direct Bedrock call: {e}")
            raise

    #Use LangChain with ChatBedrockConverse
    def langchain_converse_call(self, prompt: str, system: str) -> Dict:
        try:
            total_start = time.time()
            initial_client_id = id(self.bedrock_client)
            
            # Track import and initialization time
            init_start = time.time()
            from langchain_aws import ChatBedrockConverse
            
            chat = ChatBedrockConverse(
                model=self.model_id,
                temperature=0.7,
                max_tokens=512,
                client=self.bedrock_client
            )
            post_creation_client_id = id(chat.client)
            init_time = time.time() - init_start
            
            # Track API call time
            pre_call_client_id = id(chat.client)
            api_start = time.time()
            messages = [
                ("system", system),
                ("human", prompt)
            ]
            response = chat.invoke(messages)
            api_time = time.time() - api_start
            post_call_client_id = id(chat.client)
            
            total_time = time.time() - total_start
            
            return {
                "response": response.content,
                "latency": init_time,
                "converse_latency": api_time,
                "total_time": total_time,
                "client_reused": initial_client_id == post_creation_client_id,
                "converse_client_reused": pre_call_client_id == post_call_client_id,
                "timing_breakdown": {
                    "init": init_time,
                    "api": api_time,
                    "total": total_time
                }
            }
        except Exception as e:
            logger.error(f"Error in Langchain call: {e}")
            raise

    #Use LangChain with ChatBedrock
    def langchain_chatbedrock_converse_call(self, prompt: str, system: str) -> Dict:
        try:
            total_start = time.time()
            initial_client_id = id(self.bedrock_client)
            
            # Track import time
            import_start = time.time()
            from langchain_aws import ChatBedrock
            import_time = time.time() - import_start
            
            # Track initialization time
            init_start = time.time()
            chat = ChatBedrock(
                model=self.model_id,
                temperature=0.7,
                max_tokens=512,
                client=self.bedrock_client,
                beta_use_converse_api=True
            )
            post_creation_client_id = id(chat.client)
            init_time = time.time() - init_start
            
            # Track API call time
            pre_call_client_id = id(chat.client)
            api_start = time.time()
            messages = [
                ("system", system),
                ("human", prompt)
            ]
            response = chat.invoke(messages)
            api_time = time.time() - api_start
            post_call_client_id = id(chat.client)
            
            total_time = time.time() - total_start
            
            return {
                "response": response.content,
                "latency": init_time + import_time,
                "converse_latency": api_time,
                "total_time": total_time,
                "client_reused": initial_client_id == post_creation_client_id,
                "converse_client_reused": pre_call_client_id == post_call_client_id,
                "timing_breakdown": {
                    "import": import_time,
                    "init": init_time,
                    "api": api_time,
                    "total": total_time
                }
            }
        except Exception as e:
            logger.error(f"Error in ChatBedrock call: {e}")
            raise

    def run_comparison_tests(self, iterations: int = 5) -> None:
        results = {
            "direct": {
                "latencies": [],
                "converse_latencies": [],
                "responses": [],
                "client_reuse": [],
                "converse_client_reuse": []
            },
            "langchain": {
                "latencies": [],
                "converse_latencies": [],
                "responses": [],
                "client_reuse": [],
                "converse_client_reuse": []
            },
            "chatbedrock": {
                "latencies": [],
                "converse_latencies": [],
                "responses": [],
                "client_reuse": [],
                "converse_client_reuse": []
            }
        }
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            for i in range(iterations):
                logger.info(f"=== Iteration {i+1}/{iterations} ===")
                
                for test_case in self.test_cases:
                    # Direct Bedrock test
                    logger.info(f"Direct Bedrock API - Testing: {test_case['name']}")
                    direct_result = self.direct_bedrock_call(
                        test_case["prompt"],
                        test_case["system"]
                    )
                    results["direct"]["latencies"].append(direct_result["latency"])
                    results["direct"]["converse_latencies"].append(direct_result["converse_latency"])
                    results["direct"]["responses"].append(direct_result["response"])
                    results["direct"]["client_reuse"].append(direct_result["client_reused"])
                    results["direct"]["converse_client_reuse"].append(direct_result["converse_client_reused"])
                    
                    # Langchain test
                    logger.info(f"Langchain ChatBedrockConverse API - Testing: {test_case['name']}")
                    langchain_result = self.langchain_converse_call(
                        test_case["prompt"],
                        test_case["system"]
                    )
                    results["langchain"]["latencies"].append(langchain_result["latency"])
                    results["langchain"]["converse_latencies"].append(langchain_result["converse_latency"])
                    results["langchain"]["responses"].append(langchain_result["response"])
                    results["langchain"]["client_reuse"].append(langchain_result["client_reused"])
                    results["langchain"]["converse_client_reuse"].append(langchain_result["converse_client_reused"])
                    
                    # Langchain ChatBedrock test
                    logger.info(f"Langchain ChatBedrock API - Testing: {test_case['name']}")
                    chatbedrock_result = self.langchain_chatbedrock_converse_call(
                        test_case["prompt"],
                        test_case["system"]
                    )
                    results["chatbedrock"]["latencies"].append(chatbedrock_result["latency"])
                    results["chatbedrock"]["converse_latencies"].append(chatbedrock_result["converse_latency"])
                    results["chatbedrock"]["responses"].append(chatbedrock_result["response"])
                    results["chatbedrock"]["client_reuse"].append(chatbedrock_result["client_reused"])
                    results["chatbedrock"]["converse_client_reuse"].append(chatbedrock_result["converse_client_reused"])

            if w:
                logger.warning("Deprecation warnings found:")
                for warning in w:
                    logger.warning(str(warning.message))

            self._print_results(results)


    def _print_results(self, results: Dict) -> None:
        logger.info("\n=== Test Results ===")
        
        for method in ["direct", "langchain", "chatbedrock"]:
            logger.info(f"\n{method.capitalize()} Statistics:")
            
            if not results[method]["latencies"]:
                logger.info(f"No data collected for {method}")
                continue
                
            # Calculate average timing breakdown if available
            if "timing_breakdown" in results[method]:
                breakdowns = [r.get("timing_breakdown", {}) for r in results[method]["responses"]]
                if breakdowns:
                    logger.info("\nTiming Breakdown (averages):")
                    categories = breakdowns[0].keys()
                    for category in categories:
                        values = [b.get(category, 0) for b in breakdowns]
                        avg = statistics.mean(values)
                        logger.info(f"  {category}: {avg:.3f}s")

            # Initial call statistics
            logger.info("Initial Call:")
            latencies = results[method]["latencies"]
            if len(latencies) > 1:  # Need at least 2 points for standard deviation
                logger.info(f"  Average: {statistics.mean(latencies):.3f}")
                logger.info(f"  Median:  {statistics.median(latencies):.3f}")
                logger.info(f"  StdDev:  {statistics.stdev(latencies):.3f}")
            elif len(latencies) == 1:
                logger.info(f"  Single measurement: {latencies[0]:.3f}")
            
            # Conversation call statistics (if available)
            if "converse_latencies" in results[method] and results[method]["converse_latencies"]:
                logger.info("Conversation Call:")
                converse_latencies = results[method]["converse_latencies"]
                if len(converse_latencies) > 1:
                    logger.info(f"  Average: {statistics.mean(converse_latencies):.3f}")
                    logger.info(f"  Median:  {statistics.median(converse_latencies):.3f}")
                    logger.info(f"  StdDev:  {statistics.stdev(converse_latencies):.3f}")
                elif len(converse_latencies) == 1:
                    logger.info(f"  Single measurement: {converse_latencies[0]:.3f}")
            
            # Client reuse rates
            if results[method]["client_reuse"]:
                client_reuse_rate = sum(results[method]["client_reuse"]) / len(results[method]["client_reuse"])
                logger.info(f"Initial Client Reuse Rate: {client_reuse_rate * 100:.1f}%")
            
            if "converse_client_reuse" in results[method] and results[method]["converse_client_reuse"]:
                converse_reuse_rate = sum(results[method]["converse_client_reuse"]) / len(results[method]["converse_client_reuse"])
                logger.info(f"Conversation Client Reuse Rate: {converse_reuse_rate * 100:.1f}%")


if __name__ == "__main__":
    tester = BedrockTester()
    tester.run_comparison_tests()