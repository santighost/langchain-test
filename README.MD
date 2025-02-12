# Nova Tests: Performance Testing for AWS Bedrock Runtime

Python script for benchmarking the performance of the AWS Bedrock runtime service using different approaches.

## Project Description

This project provides a testing framework to compare the performance of direct API calls to AWS Bedrock versus using the LangChain library to interact with Bedrock. It measures initialization time, API call time, and total execution time for generating responses using the Bedrock Converse API.

The main features include:

- Direct Bedrock API calls using boto3
- LangChain integration with Bedrock using ChatBedrockConverse
- LangChain integration with Bedrock using ChatBedrock
- Performance metrics collection for each approach
- Comparison of results across multiple test cases

This tool is useful for developers and data scientists working with AWS Bedrock who want to optimize their application's performance and understand the tradeoffs between different integration approaches.

## Repository Structure

```
.
├── nova-tests.py      # Main script containing test implementations
├── README.MD          # This file
└── requirements.txt   # Python dependencies
```

## Usage Instructions

### Installation

1. Ensure you have Python 3.7+ installed
2. Clone this repository
3. Install dependencies:

```
pip install -r requirements.txt
```

4. If needed, configure credentials with access to Bedrock in your environment

```
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_SESSION_TOKEN'] = ''
```

### Running Tests

Execute the script:

```
python nova-tests.py
```

This will run a series of test cases using different approaches to interact with Bedrock and output performance metrics.

### Configuration

You can modify the `BedrockTester` class in `nova-tests.py` to adjust test parameters such as:

- Number of test iterations
- Prompt and system message content
- Bedrock model selection

### Troubleshooting

- Ensure your AWS credentials are correctly configured
- Check that you have the necessary permissions to access Bedrock
- Verify your network connection if you encounter API timeouts

## Data Flow

1. Test case initialization
2. API client creation (boto3 or LangChain)
3. Performance timer start
4. API call to Bedrock runtime
5. Response generation
6. Performance timer stop
7. Metric collection and storage
8. Results comparison and output

```
[Test Case] -> [API Client] -> [Bedrock Runtime] -> [Response] -> [Metrics]
```

## Deployment

This project is designed to be run locally or in a CI/CD pipeline for performance testing. No specific deployment is required beyond the installation steps mentioned above.