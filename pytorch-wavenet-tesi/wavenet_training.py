import torch
import torch.optim as optim
import torch.utils.data
import time
from datetime import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from model_logging import Logger
from wavenet_modules import *


def print_last_loss(opt):
    print("loss: ", opt.losses[-1])


def print_last_validation_result(opt):
    print("validation loss: ", opt.validation_results[-1])

class WavenetTrainer:
    def __init__(self,
                 model,
                 dataset,
                 optimizer=optim.Adam,
                 lr=0.001, # LEARNING RATE, TROPPO ALTO RISCHIA OVERSHOOTING, TROPPO BASSO RISCHIA DI NON IMPARARE
                 weight_decay=0,
                 gradient_clipping=None, # METODO PER EVITARE CHE I GRADIENTI CRESCANO TROPPO,
                 #POTREBBE PORTARE AD UPDATE GRANDI DEI PARAMTR E A INSTABILITà (PROBLEMI NAN O OVERFLOW) --> EXPLODING GRADIENTS
                 # CLIP FACTOR = CLIP TRESHHOLD / GRADIENT NORM, I GRADIENTI CLIPPED DIVENTANO = GRADIENT FACTOR * GRADIENTS
                 logger=None,
                 snapshot_path=None,
                 snapshot_name='snapshot',
                 snapshot_interval=1000, 
                 dtype=torch.FloatTensor,
                 ltype=torch.LongTensor):
        self.model = model      
        self.dataset = dataset
        self.dataloader = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip = gradient_clipping
        self.optimizer_type = optimizer
        # CREA INSTANZA DELL'OTTIMIZZATORE
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # INIZIALIZZIAMO CORETTAMENTE IL LOGGER
        #self.logger = logger
        self.logger = logger if logger is not None else Logger(trainer=self)
        self.snapshot_path = snapshot_path
        self.snapshot_name = snapshot_name
        self.snapshot_interval = snapshot_interval
        self.dtype = dtype
        self.ltype = ltype
        
    def train(self,
              batch_size=32,
              epochs=10,
              continue_training_at_step=0):
        self.model.train()
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      num_workers=8,
                                                      pin_memory=False)
        step = continue_training_at_step
        for current_epoch in range(epochs):
            print("epoch", current_epoch)
            tic = time.time()
            for (x, target) in iter(self.dataloader):
                                # FORWARD
                                # X DATI DI INPUT (BATCH DI CAMPIONI AUDIO)
                x = Variable(x.type(self.dtype))
                # APPIATTISCE IL TENSORE SU UN'UNICA DIMENSIONE (2, 3, 4) --> (24)
                target = Variable(target.view(-1).type(self.ltype))
                # OUTPUT DEL MODELLO: PREDIZIOINI SUL BATCH DI INPUT, TENSORE USATO PER CALCOLARE LA LOSS E AGGIORNARE I PESI
                output = self.model(x)
                # SQUEEZE RIMUOVE LE DIMENSIONI UNITARIE DEI TENSORI
                loss = F.cross_entropy(output.squeeze(), target.squeeze())
                # RESETTA GRADIENTI
                self.optimizer.zero_grad()
                                # BACKWARD
                # CALCOLA GRADIENTE DEL TENSORE CORRENTE
                loss.backward()
                #loss = loss.data[0]
                # RITORNA IL VALORE SCALARE DELLA LOSS
                loss.item()
                
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                # UPADATE PARAMETRI
                self.optimizer.step()
                step += 1
                
                # time step duration
                if step == 100:
                    toc = time.time()
                    print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")
                    
                if step % self.snapshot_interval == 0:
                    if self.snapshot_path is None:
                        continue
                    time_string =  time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
                    torch.save(self.model, self.snapshot_path + '/' + self.snapshot_name + '_' + time_string)
                # STAMPO VALORE DEL LOSS A OGNI STEP PER FARE CHECK
                print(f"[Step {step}] loss: {loss}")
                self.logger.log(step, loss)
    
    def validate(self):
        self.model.eval()
        self.dataset.train = False
        total_loss = 0
        accurate_classifications = 0
        i=1
        for (x, target) in iter(self.dataloader):
            print(f"Step {i}/{len(self.dataloader)}")
            
            x = Variable(x.type(self.dtype))
            target = Variable(target.view(-1).type(self.ltype))
            
            output = self.model(x) # OUTPUT DEL MODELLO TRAIN_SCRIPT
            loss = F.cross_entropy(output.squeeze(), target.squeeze())
            #total_loss = loss.data[0]
            total_loss += loss.item()
            # RITORNA VALORE MASSIMO E FLATTENS SU UNA DIM
            predictions = torch.max(output, 1)[1].view(-1)
            # CONFRONTA PREDIZIONI CON TARGET
            correct_pred = torch.eq(target, predictions)
            # SUM() RITORNA LA SOMMA DI TUTTI GLI ELEMENTI DEL TENSORE CORRECT_PRED
            # accurate_classifications += torch.sum(correct_pred).data[0]
            accurate_classifications += torch.sum(correct_pred).item()
            i+=1
        # print("validate model with " + str(len(self.dataloader.dataset)) + " samples")
        # print("average loss: ", total_loss / len(self.dataloader))
        # LUNGHEZZA DEL DATA LOADER == NUMERO DI BATCH, 
        # CALCOLA MEDIA DI LOSS SU TUTTI I BATCH --> MISURA COMPLESSIVA DELLA QUALITà DEL MODELLO SULLA SINGOLA EPOCH
        avg_loss = total_loss / len(self.dataloader)
        # COME PER IL LOSS MEDIO MA SUGLI ESMPI (CHE FORMANO I BATCH), CALCOLO ACCURATEZZA DELLE SINGOLE PREDIZIONI
        avg_accuracy = accurate_classifications / (len(self.dataset)*self.dataset.target_length)
        self.dataset.train = True
        self.model.train()
        return avg_loss, avg_accuracy
    
def generate_audio(model,
                   length=8000,
                   # TEMPERATURE: IPERPARAMETRO CHE CONTROLLA LA CASUALITà NELLA GENERAZIONE 
                   # (BASSA → PIù DETERMINISTICA, ALTA → PIù CREATIVA/VARIABILE) 
                   temperatures=[0.,1.]):
    # CONTERRà CAMPIONI GENERATI A TEMPERATURE DIVERSE, SONO DEGLI ARRAY 1D
    # OGNI ARRAY CONTIENE UN CLIP AUDIO DIVERSO
    samples = []
    for temp in temperatures:
        # PER OGNI TEMPERATURA GENERA CAMPIONI AUDIO DELLA LENGTH SPECIFICATA USANDO LA FAST GENERATION
        samples.append(model.generate_fast(length, temperature=temp))
        # DIMENSIONE FINALE (n_temperatures, length), 
        # UNA MATRICE IN CUI OGNI RIGA è UN CLIP AUDIO DIVERS GENERATO CON UNA TEMPERATURA DIVERSA
        # STACK() è USATO PER CONCATENARE TENSORI LUNGO UNA NUOVA DIMENSIONE
        # ESSI DEVONO AVERE STESSSA SHAPE E DIMENSIONE 
        # IL NUIOVO ARRAY AVRà UNA DIMENSIONE AGGIUNTIVA RISPETTO A QUELLI DI INPUT
        # VIENE SPECIFICATO L'ASSE LUNGO CUI FARE LO STACKING
        # AXIS = 0 INDICA CHE LA NUOVA DIMENSIONE SARA' LA PRIMA (RIGHE)
    samples = np.stack(samples, axis=0)
    return samples