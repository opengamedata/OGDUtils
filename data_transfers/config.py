settings = {
    "DATA_DIR": "./data/",
    "OGD_CORE_PATH":"Path/To/opengamedata-core",
    "MYSQL_CONFIG":
    {
        "DB_HOST" : "127.0.0.1",
        "DB_USER" : "swansonl",
        "DB_PW" : "8590-LuS",
        "DB_NAME" : "logger",
        "TABLE" : "log",
        "DB_PORT" : 3306
    },
    "SSH_CONFIG":
    {
        "SSH_HOST" : "fieldday-store.ad.education.wisc.edu",
        "SSH_USER" : "lwswanson2",
        "SSH_PW" : "GobbLer58@W7%1",
        "SSH_PORT" : 22
    },
    "BIGQUERY_CONFIG":
    {
        "AQUALAB": {
            "DB_NAME": "aqualab-57f88.analytics_271167280",
            "PROJECT_ID" : "aqualab-57f88",
            "DATASET_ID": "alt_logging_test",
            "TABLE_NAME": "opengamedata"
        },
        "MASHOPOLIS": {
            "DB_NAME": "mashopolis-36754.analytics_302498821",
            "PROJECT_ID": "mashopolis-36754",
            "DATASET_ID": "analytics_302498821"
        },
        "SHADOWSPECT": {
            "DB_NAME": "shadowspect-b8e63.analytics_284091572",
            "PROJECT_ID" : "shadowspect-b8e63",
            "DATASET_ID" : "analytics_284091572"
        },
        "SHIPWRECKS": {
            "DB_NAME": "shipwrecks-8d142.analytics_269167605",
            "PROJECT_ID" : "shipwrecks-8d142",
            "DATASET_ID" : "analytics_269167605",
        },
        "TABLE_NAME": "events_*"
    },
    "GAME_SOURCE_MAP":
    {
        "AQUALAB":{"interface":"BigQuery", "table":"BIGQUERY", "credential":"./config/aqualab.json"},
        "BACTERIA":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "BALLOON":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "CRYSTAL":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "CYCLE_CARBON":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "CYCLE_NITROGEN":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "CYCLE_WATER":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "EARTHQUAKE":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "JOWILDER":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "LAKELAND":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "MAGNET":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "SHIPWRECKS":{"interface":"BigQuery", "table":"BIGQUERY", "credential":"./config/shipwrecks.json"},
        "SHADOWSPECT":{"interface":"BigQuery", "table":"BIGQUERY", "credential":"./config/shadowspect.json"},
        "STEMPORTS":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "WAVES":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
        "WIND":{"interface":"MySQL", "table":"FIELDDAY_MYSQL", "credential":None},
    },
    "BATCH_SIZE":1000,
    "DEBUG_LEVEL":"INFO",
    "FAIL_FAST":True
}
