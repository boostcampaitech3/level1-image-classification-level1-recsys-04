from calendar import EPOCH
import wandb
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class CustomWandb:
    def __init__(self) -> None:
        self.project_name = 'Image Classification'
        self.batch_size = 32
        self.lr = 1e-4
        self.epochs = 10
        self.run_name = 'EfficientNet-Run'
    
    def config(self):
        wandb.init(
        project=self.project_name,
        config={
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            #"dropout": random.uniform(0.01, 0.80),
            })
        wandb.run.name = self.run_name

    def set_project_name(self, project_name: str) -> None:
        '''
        project name을 설정합니다.
        '''
        self.project_name = project_name

    def set_run_name(self, run_name: str) -> None:
        self.run_name = run_name
        #wandb.run.save()
    
    def set_hpppm(self, batch_size: int, lr: float, epochs: int) -> None:
        '''
        Hyper Parameter을 설정합니다.
        '''
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

    def log_3classifier_acc(self, correct_cnt: np.array, total_cnt: np.array) -> dict:
        '''
        Epoch당 Age, Mask, Gender 3개에 대한 클래스별 정확도의 Log 기록을 남깁니다.
        :correct_cnt - MultiClass Correct Count List
        :total_cnt - MultiClass Total Count List
        '''

        #idx = np.array(range(18))

        #mask_0, mask_1, mask_2 = idx[np.argwhere(idx//6%3==0)], idx[np.argwhere(idx//6%3==1)], idx[np.argwhere(idx//6%3==2)]
        #gender_0, gender_1 = idx[np.argwhere(idx==0)], idx[np.argwhere(idx==1)]
        #age_0, age_1, age_2 = idx[np.argwhere(idx%3==0)], idx[np.argwhere(idx%3==1)], idx[np.argwhere(idx%3==2)]

        #age_acc = {'age0': correct_cnt[age_0].sum() / total_cnt[age_0].sum(),
        #           'age1': correct_cnt[age_1].sum() / total_cnt[age_1].sum(),
        #           'age2': correct_cnt[age_2].sum() / total_cnt[age_2].sum()}
#
        #mask_acc = {'mask0': correct_cnt[mask_0].sum() / total_cnt[mask_0].sum(),
        #            'mask1': correct_cnt[mask_1].sum() / total_cnt[mask_1].sum(),
        #            'mask2': correct_cnt[mask_2].sum() / total_cnt[mask_2].sum()}

        gender_acc = {'gender0': correct_cnt[0] / total_cnt[0],
                      'gender1': correct_cnt[1] / total_cnt[1]}

        total = correct_cnt.sum() / total_cnt.sum()

        #age_metrics = {"Acc/age0": round(age_acc['age0']*100, 2),
        #               "Acc/age1": round(age_acc['age1']*100, 2),
        #               "Acc/age2": round(age_acc['age2']*100, 2)}
        #
        #mask_metrics =  {"Acc/mask0": round(mask_acc['mask0']*100, 2),
        #                 "Acc/mask1": round(mask_acc['mask1']*100, 2),
        #                 "Acc/mask2": round(mask_acc['mask2']*100, 2) }
        
        gender_metrics = {"Acc/gender0": round(gender_acc['gender0']*100, 2),
                          "Acc/gender1": round(gender_acc['gender1']*100, 2)}
        
        total_metrics = {"Acc/total": round(total*100, 2)}
        wandb.log({**gender_metrics, **total_metrics})

        return  gender_metrics

    
    def plot_best_model(self, correct_cnt: np.array, total_cnt: np.array) -> None:
        '''
        best 모델에 대해서 각각 Age, Gender, Mask, MultiClass에 대한 정확도를 bar plot으로 출력합니다.
        :correct_cnt - MultiClass Correct Count List
        :total_cnt - MultiClass Total Count List
        '''

        plt.rcParams['figure.dpi'] = 150  # 고해상도 설정
        colors = plt.cm.get_cmap('Pastel1').colors

        ###### Show Accuracy By 3 Classification #####
        gender = self.log_3classifier_acc(correct_cnt=correct_cnt, total_cnt=total_cnt)

        fig_3classifier, axes = plt.subplots(figsize=(15, 7))

        idx = range(len(gender))
        min_acc = min(gender.values())

        axes.bar(idx, gender.values(),
            width=0.3,
            edgecolor='black',
            linewidth=0.5,
            color=colors[0],
            zorder=10,
            align = 'center'
            )

        axes.set_xticks(idx)
        axes.set_xticklabels(list(gender.keys()))
        axes.set_ylim(min_acc-10, 100)
        axes.set_ylabel('Accuracy')

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.spines['left'].set_linewidth(1.5)
        axes.spines['bottom'].set_linewidth(1.5)

        for n, value in zip(idx, gender.values()):
            axes.text(n, value+1.5, s=f'{value}%',
                        ha='center', 
                        fontweight='bold'
                        )
        
        wandb.log({'Plot/3Classification': wandb.Image(fig_3classifier)})

        ##### Show Accuracy By MultiClassification #####
        #multi = (correct_cnt / total_cnt)*100
        #multi = np.round(multi, 2)
        #min_acc = np.min(multi)
#
        #fig_multiclassifier = plt.figure(figsize=(20, 10))
        #ax = fig_multiclassifier.add_subplot()
        #idx = range(len(multi))
#
        #ax.bar(idx, multi,
        #        width=0.5,
        #        edgecolor='black',
        #        linewidth=0.6,
        #        color=colors,
        #        zorder=10,
        #        align = 'center'
        #        )
#
        #ax.set_xlabel('MultiClass')
        #ax.set_ylabel('Accuracy')
        #ax.set_xticks(idx)
        #ax.set_ylim(min_acc-10, 100)
#
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        #ax.spines['left'].set_linewidth(1.5)
        #ax.spines['bottom'].set_linewidth(1.5)
#
        #for i, value in zip(idx, multi):
        #    ax.text(i, value+1.2, s=f'{value}%',
        #                ha='center', 
        #                fontweight='bold'
        #            )
#
        #wandb.log({'Plot/MultiClassification': wandb.Image(fig_multiclassifier)})


    def log_miss_label(self, miss_labels: list) -> None:
        '''
        Valid Dataset에서 잘못 라벨링 한 데이터의 이미지와 추론값을 표로 출력합니다.
        :miss_label - 잘못 라벨링 된 데이터의 정보를 담은 리스트, [(img, label, pred)] 형식으로 저장
        '''
        table = wandb.Table(columns=["imgs", 'label', 'pred', 'Gender'])
        
        for img, label, pred in miss_labels:
            img = img.to("cpu")
            img = wandb.Image(img) 
            #img = wandb.Image(img.numpy()*255)  
            label = label.to("cpu")
            pred = pred.to("cpu")

            #age = f'label: {label%3}  |  pred: {pred%3}'
            gender = f'label: {label}  |  pred: {pred}'
            #mask = f'label: {(label//6)%3}  |  pred: {(pred//6)%3}'

            table.add_data(img, label, pred, gender)
        
        wandb.log({"Miss_Label_table":table}, commit=False)

    def log_train_sample(self, inputs : torch.tensor, labels : torch.tensor) -> None:
        '''
        Train Dataset의 일부 데이터의 이미지와 라벨을 표로 출력합니다.
        :inputs - 이미지 정보를 담은 텐서
        :labels - 라벨 정보를 담은 텐서
        '''
        table = wandb.Table(columns=["train_imgs", 'Gender'])

        for img, label in zip(inputs.to("cpu"), labels.to("cpu")):
            img = wandb.Image(img)  
            #img = wandb.Image(img.numpy()*255)  
            table.add_data(img, label)

        wandb.log({"Train_Sample_table":table}, commit=False)
        print('finish log')


    def log(self, metric1, metric2):
        wandb.log({**metric1, **metric2})

    def finish(self):
        wandb.finish()
    




        

