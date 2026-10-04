from ipad_vad.dwell_scoring import DwellBaseline
from ipad_vad.context_dwell import ContextDwell, observed_entry_context


class ContextDwellBaseline(DwellBaseline):
    def __init__(self,config,process):
        super().__init__(config,process)
        self.dwell=ContextDwell(**config['normal_dwell'],**config['dwell_context'])

    def score(self,data):
        result=super().score(data);result['dwell_entry_context']=observed_entry_context(data)
        return result
