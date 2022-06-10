from datetime import datetime
from typing import List, Tuple

from google.cloud import bigquery_storage_v1
from google.cloud.bigquery_storage_v1 import types
from google.cloud.bigquery_storage_v1 import writer
from google.protobuf import descriptor_pb2

from data_transfers import OGDFormat_pb2
from data_transfers import CustomerRecordSample_pb2
from data_transfers.config import settings
from opengamedata.interfaces.MySQLInterface import MySQLInterface, SQL


def create_row_data_sample(row_num: int, name: str):
    """_summary_

    :param row_num: _description_
    :type row_num: int
    :param name: _description_
    :type name: str
    :return: _description_
    :rtype: _type_
    """
    row = CustomerRecordSample_pb2.CustomerRecordSample()
    row.row_num = row_num
    row.customer_name = name
    return row.SerializeToString()


def create_row_data(row_vals:Tuple) -> OGDFormat_pb2.CustomerRecord:
    """_summary_

    :param row_vals: _description_
    :type row_vals: Tuple
    :return: _description_
    :rtype: CustomerRecord
    """
    row = OGDFormat_pb2.CustomerRecord()
    row.id = row_vals[0]
    row.app_id = row_vals[1]
    row.user_id = row_vals[2]
    row.user_data = row_vals[3] #TODO: fix
    row.session_id = row_vals[4]
    row.event_name = row_vals[5]
    row.event_data = row_vals[6] #TODO: fix
    row.game_state = row_vals[7] #TODO: fix
    row.event_time = row_vals[8]
    row.event_time_ms = row_vals[9]
    row.time_offset = row_vals[10]
    row.app_version = row_vals[11]
    row.app_branch = row_vals[12]
    row.log_version = row_vals[13]
    row.http_user_agent = row_vals[14]
    row.remote_addr = row_vals[15]
    row.event_sequence_index = row_vals[16]
    return row.SerializeToString()

def append_rows_pending(project_id:str, dataset_id:str, table_id:str):
    """_summary_

    :param project_id: _description_
    :type project_id: str
    :param dataset_id: _description_
    :type dataset_id: str
    :param table_id: _description_
    :type table_id: str
    """
    write_client = bigquery_storage_v1.BigQueryWriteClient()
    parent = write_client.table_path(project_id, dataset_id, table_id)
    write_stream = types.WriteStream()
    write_stream.type_ = types.WriteStream.Type.PENDING
    write_stream = write_client.create_write_stream(
        parent=parent, write_stream=write_stream
    )
    stream_name = write_stream.name
    # Create a template with fields needed for the first request.
    request_template = types.AppendRowsRequest()
    # The initial request must contain the stream name.
    request_template.write_stream = stream_name

    # So that BigQuery knows how to parse the serialized_rows, generate a
    # protocol buffer representation of your message descriptor.
    proto_schema = types.ProtoSchema()
    proto_descriptor = descriptor_pb2.DescriptorProto()
    CustomerRecordSample_pb2.CustomerRecordSample.DESCRIPTOR.CopyToProto(proto_descriptor)
    proto_schema.proto_descriptor = proto_descriptor
    proto_data = types.AppendRowsRequest.ProtoData()
    proto_data.writer_schema = proto_schema
    request_template.proto_rows = proto_data

    # Some stream types support an unbounded number of requests. Construct an
    # AppendRowsStream to send an arbitrary number of requests to a stream.
    append_rows_stream = writer.AppendRowsStream(write_client, request_template)
    # Create a batch of row data by appending proto2 serialized bytes to the
    # serialized_rows repeated field.
    proto_rows = types.ProtoRows()
    proto_rows.serialized_rows.append(create_row_data_sample(1, "Alice"))
    proto_rows.serialized_rows.append(create_row_data_sample(2, "Bob"))



    # Set an offset to allow resuming this stream if the connection breaks.
    # Keep track of which requests the server has acknowledged and resume the
    # stream at the first non-acknowledged message. If the server has already
    # processed a message with that offset, it will return an ALREADY_EXISTS
    # error, which can be safely ignored.
    #
    # The first request must always have an offset of 0.
    request = types.AppendRowsRequest()
    request.offset = 0
    proto_data = types.AppendRowsRequest.ProtoData()
    proto_data.rows = proto_rows
    request.proto_rows = proto_data

    response_future_1 = append_rows_stream.send(request)

    # Send another batch.
    proto_rows = types.ProtoRows()
    proto_rows.serialized_rows.append(create_row_data_sample(3, "Charles"))

    # Since this is the second request, you only need to include the row data.
    # The name of the stream and protocol buffers DESCRIPTOR is only needed in
    # the first request.
    request = types.AppendRowsRequest()
    proto_data = types.AppendRowsRequest.ProtoData()
    proto_data.rows = proto_rows
    request.proto_rows = proto_data

    # Offset must equal the number of rows that were previously sent.
    request.offset = 2

    response_future_2 = append_rows_stream.send(request)

    print(response_future_1.result())
    print(response_future_2.result())

    # Shutdown background threads and close the streaming connection.
    append_rows_stream.close()

    # A PENDING type stream must be "finalized" before being committed. No new
    # records can be written to the stream after this method has been called.
    write_client.finalize_write_stream(name=write_stream.name)

    # Commit the stream you created earlier.
    batch_commit_write_streams_request = types.BatchCommitWriteStreamsRequest()
    batch_commit_write_streams_request.parent = parent
    batch_commit_write_streams_request.write_streams = [write_stream.name]
    write_client.batch_commit_write_streams(batch_commit_write_streams_request)

    print(f"Writes to stream: '{write_stream.name}' have been committed.")

mysql = MySQLInterface("AQUALAB", settings=settings)
start_date = datetime.now() # TODO: get start of yesterday.
end_date = datetime.now() # TODO: get end of yesterday
if "MYSQL_CONFIG" in settings.keys() and mysql._db_cursor is not None and mysql.IsOpen():
    db_name = settings["MYSQL_CONFIG"]["DB_NAME"]
    table_name = settings["MYSQL_CONFIG"]["TABLE"]
    rows = SQL.SELECT(cursor=mysql._db_cursor, db_name=db_name, table=table_name,
                      filter=f"`server_time` BETWEEN {start_date} AND {end_date}")
    if rows is not None and "BIGQUERY_CONFIG" in settings.keys():
        # Create a write stream, based on Google's sample code.
        _project_id = settings["BIGQUERY_CONFIG"]["AQUALAB"]["PROJECT_ID"]
        _dataset_id = settings["BIGQUERY_CONFIG"]["AQUALAB"]["DATASET_ID"]
        _table_id   = settings["BIGQUERY_CONFIG"]["TABLE_NAME"] if "TABLE_NAME" not in settings["BIGQUERY_CONFIG"]["AQUALAB"].keys() else settings["BIGQUERY_CONFIG"]["AQUALAB"]["TABLE_NAME"]
        append_rows_pending(project_id=_project_id, dataset_id=_dataset_id, table_id=_table_id)
