import json

DAQ_TASKS = [
        {
            "NAME": "Dev1",
            "SAMPLE_RATE": 10000,

            "DAQ_CHANNELS":{
                "Voltage": {"port": "Dev1/ai3", "port_config": 0, "conversion_factor": None},
                # "Current": {"port": "Dev1/ai2", "port_config": DAQmx_Val_RSE, "conversion_factor": 1},
            },

            # "TRIGGER_SOURCE": "PFI0",
            "TRIGGER_SOURCE": None,

            "TYPE": "analog"
        },

        {
            "NAME": "Dev2",
            "SAMPLE_RATE": 10000,

            "DAQ_CHANNELS":{
                # "LinMot_Enable": {"port":"Dev2/ai0", "port_config":DAQmx_Val_RSE, "conversion_factor": None},
                # "LinMot_Up_Down": {"port":"Dev2/ai1", "port_config":DAQmx_Val_RSE, "conversion_factor": None}
                "LinMot_Enable": {"port":"Dev2/port0/line0", "port_config":None, "conversion_factor": None},
                "LinMot_Up_Down": {"port":"Dev2/port0/line1", "port_config":None, "conversion_factor": None}
            },

            # "TRIGGER_SOURCE": "PFI0",
            "TRIGGER_SOURCE": None,
            "TYPE": "digital"
        }
    ]

json_str = json.dumps(DAQ_TASKS, indent=4)
with open("sample.json", "w") as f:
    f.write(json_str)