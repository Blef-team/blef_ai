import boto3
import json
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


lambda_client = boto3.client('lambda')


def parse_event(event):
    # Basic input validation
    if not isinstance(event, dict):
        return False

    # Handle both direct triggers and API Gateway
    body = event.get("body", event)
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except ValueError:
            return None
    path_params = event.get("pathParameters", {})
    query_params = event.get("queryStringParameters", {})
    body.update(path_params)
    body.update(query_params)
    return body


def get_worker_name(game):
    players = game["players"]
    hand_sizes = [player.get("n_cards") for player in players]
    hand_sizes.sort()
    return '-'.join(str(x) for x in hand_sizes)


def lambda_function_exists(name):
    try:
        response = lambda_client.get_function(FunctionName=name)
    except lambda_client.exceptions.ResourceNotFoundException:
        return False
    return response.get("ResponseMetadata").get("HTTPStatusCode") == 200


def call_worker(payload):
    """
        Invoke blef-aiagent-cfr-worker-[...] asynchronously
    """
    worker_name = get_worker_name(payload)
    logger.info("## WORKER_NAME")
    logger.info(worker_name)
    function_name = f'blef-aiagent-cfr-worker-{worker_name}'
    if lambda_function_exists(function_name):
        return lambda_client.invoke(
            FunctionName=function_name,
            InvocationType='Event',
            Payload=json.dumps(payload)
        )
    logger.info("## ASKING POREVIT")
    return lambda_client.invoke(
        FunctionName='blef-aiagent-conservative-crawling',
        InvocationType='Event',
        Payload=json.dumps(payload)
    )


def get_game(record):
    if not record.get("body"):
        return
    if not isinstance(record.get("body"), dict):
        try:
            return json.loads(record.get("body"))
        except ValueError as e:
            return


def lambda_handler(event, context):
    logger.info('## EVENT')
    logger.info(event)
    game = parse_event(event)
    logger.info('## GAME')
    logger.info(game)
    assert game

    call_worker(game)
    message = "Messages processed"
    return {
        'statusCode': 200,
        'body': json.dumps({"message": message})
    }
