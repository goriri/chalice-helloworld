import logging
from chalice import Chalice
import boto3
import pandas as pd

app = Chalice(app_name='helloworld')
app.log.setLevel(logging.DEBUG)

client = boto3.client('comprehend')
sns_client = boto3.client('sns')
s3 = boto3.client('s3')


@app.on_s3_event(bucket='chalice-test-348052051973',
                 events=['s3:ObjectCreated:*'], suffix='csv')
def invoke_comprehend(event):
    app.log.debug("Received event for bucket: %s, key: %s",
                  event.bucket, event.key)
    s3uri = 's3://' + event.bucket + '/' + event.key
    app.log.debug("s3uri: %s", s3uri)
    response = client.start_document_classification_job(
        JobName='test',
        DocumentClassifierArn='arn:aws:comprehend:us-east-1:348052051973:document-classifier/ScamModel/version/2',
        InputDataConfig={
            'S3Uri': s3uri,
            'InputFormat': 'ONE_DOC_PER_LINE',
        },
        OutputDataConfig={
            'S3Uri': 's3://chalice-test-output-348052051973',
        },
        DataAccessRoleArn='arn:aws:iam::348052051973:role/service-role/AmazonComprehendServiceRoleS3FullAccess-ComprehendLabs',
    )
    app.log.debug("response: %s", response)


@app.on_s3_event(bucket='chalice-test-output-348052051973',
                 events=['s3:ObjectCreated:*'], suffix='output.tar.gz')
def notify_comprehend_done_v2(event):
    app.log.debug("Received event for bucket: %s, key: %s",
                  event.bucket, event.key)
    s3uri = 's3://' + event.bucket + '/' + event.key

    prediction_df = pd.read_json(s3uri, compression='gzip', error_bad_lines=False, lines=True)
    prediction_df = prediction_df.set_index("Line")
    prediction_df = prediction_df.drop(columns=["File"])
    app.log.debug("prediction: %s", prediction_df.head(10))

    response = sns_client.publish(
        TopicArn='arn:aws:sns:us-east-1:348052051973:MyTopic',
        Message=prediction_df.to_json(),
        MessageStructure='json',
        Subject=f'{s3uri} has new result',
    )
    app.log.debug("response: %s", response)

# The view function above will return {"hello": "world"}
# whenever you make an HTTP GET request to '/'.
#
# Here are a few more examples:
#
# @app.route('/hello/{name}')
# def hello_name(name):
#    # '/hello/james' -> {"hello": "james"}
#    return {'hello': name}
#
# @app.route('/users', methods=['POST'])
# def create_user():
#     # This is the JSON body the user sent in their POST request.
#     user_as_json = app.current_request.json_body
#     # We'll echo the json body back to the user in a 'user' key.
#     return {'user': user_as_json}
#
# See the README documentation for more examples.
#