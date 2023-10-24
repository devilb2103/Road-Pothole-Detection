import streamlit

def initializeSessionVariables(st: streamlit):
    stateVars = {
        "DataCollectionStage":True,
        "CameraInput":False,
    }
    for i in stateVars.items():
        if(i[0] not in st.session_state):
            st.session_state[i[0]] = i[1]