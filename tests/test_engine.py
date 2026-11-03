import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from mfde import MFDEEngine, PriorPreservationTeacher
from mfde.multi_engine import classification_metrics, unpack_batch
from mfde.objective import compute_objective
from train import build_model, run


class Bags(Dataset):
    def __init__(self, count):
        self.items = [({'uni': torch.randn(5+i, 8), 'phikon': torch.randn(5+i, 12)}, i % 2) for i in range(count)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class ScriptedEngine(MFDEEngine):
    def train(self, loader):
        self.current = getattr(self, 'current', -1) + 1
        with torch.no_grad():
            self.model.weight.fill_(self.current + 1)
        return {'total': 1.0}

    def validate(self, loader, model=None):
        self.calls.append(loader)
        values = [.9, .7, .8] if loader == 'val' else [.6, .95, .7]
        return {'roc_auc': values[self.current]}

    def testing(self, loader, model=None, output_name='target_id_outputs.pt'):
        self.calls.append(output_name)
        value = [.6, .95, .7][int(self.model.weight.item()) - 1]
        return {'roc_auc': value}


class EngineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.config = {'embed_dim':16,'num_classes':2,'attention_heads':4,'attention_dim':8,'attention_dropout':0.0,'num_prototypes':4,'ema_decay':0.9,'epochs':1,'virtual_batch_size':2,'engine':{'export_predictions':True,'export_attention':True},'objective':{'prior_implementation':'manuscript_reference','routing_temperature':0.1,'lambda_aux':0.5,'lambda_route':1,'lambda_reg_effective':1}}

    def test_engine_accumulation_ema_and_attention_export(self):
        train_loader=DataLoader(Bags(3),batch_size=1)
        evaluation=DataLoader(Bags(2),batch_size=1)
        sample,_=train_loader.dataset[0]
        model=build_model(self.config,sample)
        expected=copy.deepcopy(model)
        expected_teacher=PriorPreservationTeacher(expected,decay=self.config['ema_decay'])
        optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
        expected_optimizer=torch.optim.SGD(expected.parameters(),lr=0.001)
        with tempfile.TemporaryDirectory() as tmp:
            engine=MFDEEngine(model,optimizer,None,self.config,tmp,logger=lambda row:None)
            engine.train(train_loader)
            expected.train()
            pending=[]
            for i,(features,labels) in enumerate(train_loader):
                loss,_=compute_objective(expected(features),expected_teacher(features),labels,expected,self.config['objective'])
                pending.append(loss)
                if len(pending)==2 or i==2:
                    torch.stack(pending).mean().backward()
                    expected_optimizer.step()
                    expected_optimizer.zero_grad()
                    expected_teacher.update(expected)
                    pending=[]
            self.assertEqual(engine.optimizer_updates,2)
            for key,value in expected.state_dict().items():torch.testing.assert_close(value,model.state_dict()[key])
            for key,value in expected_teacher.state_dict().items():torch.testing.assert_close(value,engine.teacher.state_dict()[key])
            metrics=engine.testing(evaluation)
            exported=torch.load(Path(tmp)/'target_id_outputs.pt',weights_only=True)
            self.assertTrue(0<=metrics['roc_auc']<=1)
            self.assertEqual(tuple(exported['routing_weights'].shape),(2,2))
            self.assertEqual(tuple(exported['attention'][0]['head_scores'].shape),(2,4,5))
            self.assertEqual(tuple(exported['attention'][1]['instance_weights'].shape),(2,6))
            torch.testing.assert_close(exported['attention'][0]['instance_weights'].sum(-1),torch.ones(2))

    def test_separate_id_oracle_selection_and_saved_weights(self):
        for oracle in [False,True]:
            config=copy.deepcopy(self.config)
            config['epochs']=3
            config['engine']['oracle_reporting']=oracle
            model=torch.nn.Linear(1,1,bias=False)
            with tempfile.TemporaryDirectory() as tmp:
                engine=ScriptedEngine(model,torch.optim.SGD(model.parameters(),lr=.1),None,config,tmp,logger=lambda row:None)
                engine.calls=[]
                result=engine.learning('train','val','test')
                self.assertEqual(result['selected_epoch'],0)
                self.assertEqual(float(model.weight),1)
                self.assertEqual(result['target_auc'],.6)
                self.assertEqual(engine.calls.count('test'),3 if oracle else 0)
                self.assertEqual(engine.calls.count('target_id_outputs.pt'),1)
                if oracle:
                    self.assertEqual(result['oracle']['selected_epoch'],1)
                    saved=torch.load(Path(tmp)/'best_oracle.pt',weights_only=True)
                    self.assertEqual(float(saved['weight']),2)
                    self.assertEqual(result['oracle']['target_auc'],.95)
                else:
                    self.assertNotIn('oracle',result)
                    self.assertFalse((Path(tmp)/'best_oracle.pt').exists())

    def test_early_stopping_uses_source_validation(self):
        config=copy.deepcopy(self.config)
        config['epochs']=3
        config['engine'].update(oracle_reporting=True,early_stopping_patience=1)
        model=torch.nn.Linear(1,1,bias=False)
        with tempfile.TemporaryDirectory() as tmp:
            engine=ScriptedEngine(model,torch.optim.SGD(model.parameters(),lr=.1),None,config,tmp,logger=lambda row:None)
            engine.calls=[]
            result=engine.learning('train','val','test')
            self.assertEqual(result['epochs_completed'],2)
            self.assertEqual(result['selected_epoch'],0)
            self.assertEqual(result['oracle']['selected_epoch'],1)

    def test_averaged_checkpoint_matches_evaluated_weights(self):
        config=copy.deepcopy(self.config)
        config['engine']['parameter_averaging_start_update']=1
        train_loader=DataLoader(Bags(3),batch_size=1)
        validation=DataLoader(Bags(2),batch_size=1)
        sample,_=train_loader.dataset[0]
        model=build_model(config,sample)
        with tempfile.TemporaryDirectory() as tmp:
            engine=MFDEEngine(model,torch.optim.SGD(model.parameters(),lr=.002),None,config,tmp,logger=lambda row:None)
            engine.learning(train_loader,validation,validation)
            self.assertEqual(int(engine.averaged_model.n_averaged),2)
            saved=torch.load(Path(tmp)/'best_source_validation.pt',weights_only=True)
            for key,value in engine.averaged_model.module.state_dict().items():torch.testing.assert_close(value,saved[key])

    def test_routing_with_auxiliary_disabled_and_legacy_batch(self):
        sample={'uni':torch.randn(1,5,8),'phikon':torch.randn(1,5,12)}
        model=build_model(self.config,sample)
        teacher=PriorPreservationTeacher(model)
        config={**self.config['objective'],'lambda_aux':0,'lambda_reg_effective':0}
        student=model(sample)
        labels=torch.tensor([1])
        loss,metrics=compute_objective(student,teacher(sample),labels,model,config)
        aux=student['aux_logits'].detach().softmax(-1)
        target=(aux[:,:,1]/.1).softmax(-1)
        expected=F.cross_entropy(student['logits'],labels)+F.kl_div((student['weights']+1e-8).log(),target,reduction='batchmean')
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertEqual(metrics['prior'],0)
        legacy=(['case'],['slide'],sample,labels,torch.zeros(1,5,2),torch.tensor([0]))
        self.assertIs(unpack_batch(legacy)[0],sample)

    def test_multiclass_metrics_and_undefined_auc(self):
        metrics=classification_metrics([0,1,2],[[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]],3)
        self.assertEqual(metrics['roc_auc'],1)
        self.assertEqual(metrics['average_precision'],1)
        with self.assertRaisesRegex(ValueError,'every configured class'):
            classification_metrics([0,0],[[.8,.2],[.6,.4]],2)

    def test_prediction_export_rejects_ambiguous_slide_order(self):
        import pandas as pd
        data=Bags(2)
        data.frame=pd.DataFrame({'ID':['first','second']})
        loader=DataLoader(data,batch_size=1,shuffle=True)
        model=build_model(self.config,data[0][0])
        with tempfile.TemporaryDirectory() as tmp:
            engine=MFDEEngine(model,torch.optim.SGD(model.parameters(),lr=.001),None,self.config,tmp)
            with self.assertRaisesRegex(ValueError,'sequential evaluation'):
                engine.testing(loader)

    def test_cli_run_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            records=['Case,ID,Label,DomainLabel,Split']
            roots={name:root/name for name in ['uni','phikon']}
            for directory in roots.values():directory.mkdir()
            for split,count in [('train',3),('val',2),('test',2)]:
                for i in range(count):
                    name=f'{split}_{i}'
                    domain='target' if split=='test' else 'source'
                    records.append(f'{name},{name},{i%2},{domain},{split}')
                    for encoder,dim in [('uni',8),('phikon',12)]:torch.save(torch.randn(5+i,dim),roots[encoder]/f'{name}.pt')
            split_csv=root/'splits.csv';split_csv.write_text('\n'.join(records),encoding='utf-8')
            config={**copy.deepcopy(self.config),'dataset':'synthetic','task':'source_to_target','seed':42,'split_csv':str(split_csv),'feature_dirs':{k:str(v) for k,v in roots.items()},'output_dir':str(root/'run'),'device':'cpu','optimizer':'Adam','lr':.0002,'weight_decay':.00001,'implementation':'synthetic_engine_test'}
            config['engine']['oracle_reporting']=True
            with contextlib.redirect_stdout(io.StringIO()):result=run(config)
            self.assertEqual(result['selection'],'source_validation')
            self.assertIn('oracle_target_auc',result)
            details=json.loads((root/'run/engine_result.json').read_text())
            self.assertEqual(details['optimizer_updates'],2)
            self.assertEqual(details['oracle']['selection'],'target_oracle')
            self.assertTrue((root/'run/target_id_outputs.pt').exists())
            self.assertTrue((root/'run/target_oracle_outputs.pt').exists())


if __name__=='__main__':
    unittest.main()
