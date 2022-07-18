import logging
from chalice import Chalice
import boto3
import pandas as pd
import json

app = Chalice(app_name='helloworld')
app.log.setLevel(logging.DEBUG)

client = boto3.client('comprehend')
sns_client = boto3.client('sns')
s3 = boto3.client('s3')


@app.on_s3_event(bucket='ab2-fraud-detection-348052051973',
                 events=['s3:ObjectCreated:*'], prefix='input', suffix='csv')
def trigger_detection(event):
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


@app.on_s3_event(bucket='ab2-fraud-detection-348052051973',
                 events=['s3:ObjectCreated:*'], prefix='output', suffix='predictions.jsonl')
def handle_fraud_detection_result(event):
    app.log.debug("Received event for bucket: %s, key: %s",
                  event.bucket, event.key)
    s3uri = 's3://' + event.bucket + '/' + event.key

    prediction_df = pd.read_json(s3uri, lines=True)
    prediction_df = prediction_df.set_index("Line")
    filename = prediction_df["File"][0]
    prediction_df = prediction_df.drop(columns=["File"])

    # app.log.debug("prediction: %s", prediction_df.head(10))
    result = pd.concat([prediction_df, prediction_df['Classes'].apply(json_to_series)], axis=1)
    result.loc[result["1"]>0.5, ["fraud"]] = True
    result = result.fillna(False)
    fraud_count = result.loc[result["fraud"] == True].shape[0]

    response = sns_client.publish(
        TopicArn='arn:aws:sns:us-east-1:348052051973:MyTopic',
        Message=f'For more details, please go to {s3uri}',
        Subject=f'{filename} has {fraud_count} fraud detected',
    )
    app.log.debug("response: %s", response)

def json_to_series(labels):
    keys, values = zip(*[(label["Name"], label["Score"]) for label in labels])
    return pd.Series(values, index=keys)
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
