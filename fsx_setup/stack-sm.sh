#!/bin/bash

# AWS Region; customize as needed
if [ -z "$1" ]
  then
    "AWS_REGION is required" && exit 1
fi
AWS_REGION=$1
echo "AWS_REGION=${AWS_REGION}"

if [ -z "$2" ]
  then
    "S3_BUCKET is required" && exit 1
fi
S3_BUCKET=$2
echo "S3_BUCKET=${S3_BUCKET}"

DATE=`date +%s`

#Customize stack name as needed
STACK_NAME="sm-stack-$DATE"

# customzie note book instance name as needed
#NOTEBOOK_INSTANCE_NAME="sm-nb-$DATE"

# cfn template name
CFN_TEMPLATE='cfn-sm.yaml'

# Leave blank if you need to create a new EFS file system
# If you use an existing EFS file-system, it must not have any
# existing mount targets
EFS_ID=

# Notebook instance type
# ml.m5.2xlarge or ml.m4.2xlarge
#NOTEBOOK_INSTANCE_TYPE='ml.m5.2xlarge'

# Code repository name
CODE_REPO_NAME="code-repo-$DATE"

# Git hub user name
GIT_USER=

# Git hub token
GIT_TOKEN=

# Git Hub repo url
GIT_URL=

# EBS volume size 100 - 500 GB
EBS_VOLUME_SIZE=100

aws cloudformation create-stack --region $AWS_REGION  --stack-name $STACK_NAME \
--template-body file://$CFN_TEMPLATE \
--capabilities CAPABILITY_NAMED_IAM \
--parameters \
ParameterKey=S3BucketName,ParameterValue=$S3_BUCKET

echo "Creating stack [ eta 600 seconds ]"
sleep 30

progress=$(aws --region $AWS_REGION cloudformation list-stacks --stack-status-filter 'CREATE_IN_PROGRESS' | grep $STACK_NAME | wc -l)
while [ $progress -ne 0 ]; do
let elapsed="`date +%s` - $DATE"
echo "Stack $STACK_NAME status: Create in progress: [ $elapsed secs elapsed ]"
sleep 30
progress=$(aws --region $AWS_REGION  cloudformation  list-stacks --stack-status-filter 'CREATE_IN_PROGRESS' | grep $STACK_NAME | wc -l)
done
sleep 5
aws --region $AWS_REGION  cloudformation describe-stacks --stack-name $STACK_NAME