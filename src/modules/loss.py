import warnings
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import complete_box_iou_loss as ciou_loss
import math


"""
def compute_HW_loss(self, predicted_states, indices, targets):

    batch_size, _ = predicted_states.shape[:2]
    loss_function = nn.MSELoss()
    loss = 0
    for i in range(batch_size):
        pred_HW = predicted_states[i][indices[i][0],-2:]
        targ_HW = targets[i][indices[i][1],-2:]
        loss += loss_function(pred_HW, targ_HW)

    return loss
"""

def check_gospa_parameters(c, p, alpha):
    """ Check parameter bounds.

    If the parameter values are outside the allowable range specified in the
    definition of GOSPA, a ValueError is raised.
    """
    if alpha <= 0 or alpha > 2:
        raise ValueError("The value of alpha is outside the range (0, 2]")
    if c <= 0:
        raise ValueError("The cutoff distance c is outside the range (0, inf)")
    if p < 1:
        raise ValueError("The order p is outside the range [1, inf)")


class MotLoss(nn.Module):
    def __init__(self, params):
        super().__init__()
        if params.loss.type == 'gospa':
            check_gospa_parameters(params.loss.cutoff_distance, params.loss.order, params.loss.alpha)
            self.order = params.loss.order
            self.cutoff_distance = params.loss.cutoff_distance
            self.alpha = params.loss.alpha
            self.miss_cost = self.cutoff_distance ** self.order
        self.params = params
        self.device = torch.device(params.training.device)
        self.to(self.device)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                              for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


    def compute_BB_loss(self, predicted_states, predicted_bbs, targets, indices):
        
        batch_size, _ = predicted_states.shape[:2]
        loss = 0
        
        for i in range(batch_size):
            
            targ_corners1 = torch.cat((torch.ones_like(targets[i])*10, torch.ones_like(targets[i])*10 + targets[i]),-1)
            #targ_corners1 = self.correct_coordinates(targ_corners1) 
            pred_corners1 = torch.cat((torch.ones_like(targets[i])*10, torch.ones_like(targets[i])*10 + predicted_bbs[i][indices[i][0]]), -1)
            #pred_corners1 = self.correct_coordinates(pred_corners1) 

            loss += ciou_loss(pred_corners1, targ_corners1, reduction='sum')

            #targ_corners2 = torch.cat((targets[i][indices[i][1], 2:4], targets[i][indices[i][1], 4:6]),-1)
            #targ_corners2 = self.correct_coordinates(targ_corners2)

            #pred_corners2 = torch.cat((predicted_states[i][indices[i][0], 2:4], predicted_states[i][indices[i][0], 4:6]),-1)
            #pred_corners2 = self.correct_coordinates(pred_corners2) 

            #loss += ciou_loss(pred_corners2, targ_corners2, reduction='sum')

        return loss
    
    def coordinates_loss(self, predicted_states, targets, indices):
        mse_loss = nn.MSELoss()
        batch_size, _ = predicted_states.shape[:2]
        loss = 0
        for i in range(batch_size):
            preds = predicted_states[i][indices[i][0]][:,:-2]
            targs = targets[i][indices[i][1]][:,:-2]
            loss += mse_loss(preds,targs)

        return loss

    def compute_hungarian_matching(self, predicted_states, predicted_bbs, predicted_logits, targets, targets_bb, distance='detr', scaling=1):
        """ Performs the matching

        Params:
            outputs: dictionary with 'state' and 'logits'
            state: Tensor of dim [batch_size, num_queries, d_label]
            logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs, num_queries = predicted_states.shape[:2]
        predicted_probabilities = predicted_logits.sigmoid()
        targets_HM=[]
        predicted_states_HM=[]

        for i in range(bs):
            pred=torch.cat((predicted_states[i], predicted_bbs[i]),dim=1)
            targ=torch.cat((targets[i], targets_bb[i]),dim=1) #x,y,vx,vy,H,B #kanske bara behöver x,y,H,B? labels_pv[i][0:1]typ?
            targets_HM.append(targ)
            predicted_states_HM.append(pred)

        targets=targets_HM
        predicted_states=predicted_states_HM

        indices = []
        for i in range(bs):
            # Compute cost matrix for this batch position
            cost = torch.cdist(predicted_states[i], targets[i], p=2)
            cost -= predicted_probabilities[i].log()

            # Compute minimum cost assignment and save it
            with torch.no_grad():
                indices.append(linear_sum_assignment(cost.cpu()))

        permutation_idx = [(torch.as_tensor(i, dtype=torch.int64).to(self.device),
                            torch.as_tensor(j, dtype=torch.int64).to(self.device)) for i, j in indices] #indices

        return permutation_idx, cost.to(self.device)

    def gospa_forward(self, outputs, targets, probabilistic=True, existence_threshold=0.75):

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"

        output_state = outputs['state']
        output_logits = outputs['logits'].sigmoid()
        # List with num_objects for each training-example
        sizes = [len(v) for v in targets]

        bs = output_state.shape[0]
        if probabilistic:
            indices, cost_matrix, unmatched_x = self.compute_prob_gospa_matching(outputs, targets)
            cost_matrix = cost_matrix.split(sizes, -1)
            loss = 0
            for i in range(bs):
                batch_idx = indices[i]
                batch_cost = cost_matrix[i][i][batch_idx].sum()
                batch_cost = batch_cost + output_logits[i][unmatched_x[i]].sum() * self.miss_cost/2.0
                loss = loss + batch_cost
            loss = loss/sum(sizes)
            return loss, indices
        else:
            assert 0 <= existence_threshold < 1, "'existence_threshold' should be in range (0,1)"
            loss, indices, decomposition = self.compute_orig_gospa_matching(outputs, targets, existence_threshold)
            loss = loss / bs
            return loss, indices, decomposition

    def state_loss(self, predicted_states, targets, indices, uncertainties=None):
        idx = self._get_src_permutation_idx(indices)
        matched_predicted_states = predicted_states[idx]
        target = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)
        if uncertainties is not None:
            matched_uncertainties = uncertainties[idx]
            prediction_distribution = torch.distributions.normal.Normal(matched_predicted_states, matched_uncertainties)
            loss = -prediction_distribution.log_prob(target).mean()
        else:
            loss = F.l1_loss(matched_predicted_states, target, reduction='none').sum(-1).mean()

        return loss

    def logits_loss(self, predicted_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        
        target_classes = torch.zeros_like(predicted_logits, device=predicted_logits.device)
        target_classes[idx] = 1.0  # this is representation of an object
        loss = F.binary_cross_entropy_with_logits(predicted_logits.squeeze(-1).permute(1,0), target_classes.squeeze(-1).permute(1,0))

        return loss

    def get_loss(self, prediction, targets_pv, targets_bb, loss_type, existence_threshold=None):
        # Create state vectors for the predictions, based on prediction target specified by user
        if self.params.data_generation.prediction_target == 'position':
            predicted_states = prediction.positions
        elif self.params.data_generation.prediction_target == 'position_and_velocity':
            predicted_states = torch.cat((prediction.positions, prediction.velocities), dim=2)
            # Why not dis instead: predicted_states = prediction.states
        elif self.params.data_generation.prediction_target == 'pos_vel_bbs':
            predicted_states = prediction.states
            predicted_bbs = prediction.bounding_boxes
        else:
            raise NotImplementedError(f'Hungarian matching not implemented for prediction target '
                                      f'{self.params.data_generation.prediction_target}')

        if loss_type == 'gospa':
            loss, indices = self.gospa_forward(prediction, targets_pv, probabilistic=True)
            loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': 0}
        elif loss_type == 'gospa_eval':
            loss,_ = self.gospa_forward(prediction, targets_pv, probabilistic=False, existence_threshold=existence_threshold)
            indices = None
            loss = {f'{loss_type}_state': loss, f'{loss_type}_logits': 0}
        elif loss_type == 'detr':
            indices, _ = self.compute_hungarian_matching(predicted_states, predicted_bbs, prediction.logits, targets_pv, targets_bb)
            log_loss = self.logits_loss(prediction.logits, targets_pv, indices)
            #HW_loss = self.compute_HW_loss(predicted_states, indices, targets)
            #coords_loss = self.coordinates_loss(predicted_states, targets, indices)
            BB_loss = self.compute_BB_loss(predicted_states, predicted_bbs, targets_bb, indices )


            if hasattr(prediction, 'uncertainties'):
                state_loss = self.state_loss(predicted_states, targets_pv, indices, uncertainties=prediction.uncertainties)
            else:
                state_loss = self.state_loss(predicted_states, targets_pv, indices)
            loss = {f'{loss_type}_state': state_loss, f'{loss_type}_logits': log_loss,f'{loss_type}_BB': BB_loss}
            #loss = {f'{loss_type}_state': state_loss, f'{loss_type}_logits': log_loss}
        
        return loss, indices
    
    def forward(self, targets_pv, targets_bb, prediction, intermediate_predictions=None, encoder_prediction=None, loss_type='detr',
                existence_threshold=0.75):
        if loss_type not in ['gospa', 'gospa_eval', 'detr']:
            raise NotImplementedError(f"The loss type '{loss_type}' was not implemented.'")

        losses = {}
        loss, indices = self.get_loss(prediction, targets_pv, targets_bb, loss_type, existence_threshold) ##Loss för sista pred
        losses.update(loss)

        if intermediate_predictions is not None: ##Loss för intermediate pred
            for i, intermediate_prediction in enumerate(intermediate_predictions):
                aux_loss, _ = self.get_loss(intermediate_prediction, targets_pv, targets_bb, loss_type, existence_threshold)
                aux_loss = {f'{k}_{i}': v for k, v in aux_loss.items()}
                losses.update(aux_loss)

        if encoder_prediction is not None: ##Loss för encoder preds
            enc_loss, _ = self.get_loss(encoder_prediction, targets_pv, targets_bb, loss_type, existence_threshold)
            enc_loss = {f'{k}_enc': v for k, v in enc_loss.items()}
            losses.update(enc_loss)

        return losses, indices #Massor dependencies i denna lol, ha så kul

    '''def compute_orig_gospa_matching(self, outputs, targets, existence_threshold): #GOSPAN ÄR HÄR
        """ Performs the matching. Note that this can NOT be used as a loss function

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

            existence_threshold: Float in range (0,1) that decides which object are considered alive and which are not.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"
        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        output_state = outputs['state'].detach()
        output_existence_probabilities = outputs['logits'].sigmoid().detach()

        bs, num_queries = output_state.shape[:2] #Bs=batches
        dim_predictions = output_state.shape[2]
        dim_targets = targets[0].shape[1]
        assert dim_predictions == dim_targets

        loss = torch.zeros(size=(1,))
        localization_cost = 0
        missed_target_cost = 0
        false_target_cost = 0
        indices = []

        for i in range(bs): #Ställer upp en distansmatris för varje batch
            alive_idx = output_existence_probabilities[i, :].squeeze(-1) > existence_threshold #Gatear logitsen
            alive_output = output_state[i, alive_idx, :]
            current_targets = targets[i]
            permutation_length = 0

            if len(current_targets) == 0: #Finns inga i labels, inget avstånd mäts
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(alive_output)])
                false_target_cost = self.miss_cost/self.alpha * len(alive_output)
            elif len(alive_output) == 0: #Finns inga i preds, inget avstånd mäts
                indices.append(([], []))
                loss += torch.Tensor([self.miss_cost/self.alpha * len(current_targets)])
                missed_target_cost = self.miss_cost / self.alpha * len(current_targets)
            else: #Mäter avstånd
                dist = torch.cdist(alive_output, current_targets, p=2) #Ställer upp distansmatris mellan sanna logits och objekt
                dist = dist.clamp_max(self.cutoff_distance)
                c = torch.pow(input=dist, exponent=self.order)
                c = c.cpu()
                output_idx, target_idx = linear_sum_assignment(c) #Väljer ut och matchar input med output
                indices.append((output_idx, target_idx))

                for t, o in zip(output_idx, target_idx): #Kollar i distansmatrisen
                    loss += c[t,o]
                    if c[t, o] < self.cutoff_distance:
                        localization_cost += c[t, o].item()
                        permutation_length += 1
                
                cardinality_error = abs(len(alive_output) - len(current_targets))
                loss += self.miss_cost/self.alpha * cardinality_error

                missed_target_cost += (len(current_targets) - permutation_length) * (self.miss_cost/self.alpha)
                false_target_cost += (len(alive_output) - permutation_length) * (self.miss_cost/self.alpha)

        decomposition = {'localization': localization_cost, 'missed': missed_target_cost, 'false': false_target_cost,
                         'n_matched_objs': permutation_length}
        return loss, indices, decomposition'''
    
    def compute_orig_gospa_matching_EOT(self, outputs, labels_pv, labels_bb, existence_threshold): #GOSPA EOT ÄR HÄR
        """ Performs the matching. Note that this can NOT be used as a loss function

        Params:
            outputs: dictionary with 'state' and 'logits'
                state: Tensor of dim [batch_size, num_queries, d_label]
                logits: Tensor of dim [batch_size, num_queries, number_of_classes]

            targets: This is a list of targets (len(targets) = batch_size), where each target is a
                    tensor of dim [num_objects, d_label] (where num_objects is the number of ground-truth
                    objects in the target)

            existence_threshold: Float in range (0,1) that decides which object are considered alive and which are not.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        assert 'state' in outputs, "'state' should be in dict"
        assert 'logits' in outputs, "'logits' should be in dict"
        assert self.order == 1, 'This code does not work for loss.order != 1'
        assert self.alpha == 2, 'The permutation -> assignment relation used to decompose GOSPA might require that loss.alpha == 2'

        output_state = outputs['state'].detach()
        output_existence_probabilities = outputs['logits'].sigmoid().detach()

        bs, num_queries = output_state.shape[:2] #Bs=batches
        dim_predictions = output_state.shape[2]
        targets=[]
        for i in range(bs):
            targ=torch.cat((labels_pv[i], labels_bb[i]),dim=1) #x,y,vx,vy,H,B #kanske bara behöver x,y,H,B? labels_pv[i][0:1]typ?
            targets.append(targ)
        dim_targets = targets[0].shape[1]
        assert dim_predictions == dim_targets

        indices = []
        loc=0
        miss=0
        false=0
        permutation=0
        loss = torch.zeros(size=(1,))

        for i in range(bs): #Ställer upp en distansmatris för varje batch
            alive_idx = output_existence_probabilities[i, :].squeeze(-1) > existence_threshold #Gatear logitsen
            alive_output = output_state[i, alive_idx, :]
            current_targets = targets[i]
            permutation_length = 0
            loss_cost = 0
            localization_cost = 0
            missed_target_cost = 0
            false_target_cost = 0

            if len(current_targets) == 0: #Finns inga i labels, inget avstånd mäts
                indices.append(([], []))
                loss_cost += torch.Tensor([self.miss_cost/self.alpha * len(alive_output)])
                false_target_cost += self.miss_cost/self.alpha * len(alive_output)
            elif len(alive_output) == 0: #Finns inga i preds, inget avstånd mäts
                indices.append(([], []))
                loss_cost += torch.Tensor([self.miss_cost/self.alpha * len(current_targets)])
                missed_target_cost += self.miss_cost/self.alpha * len(current_targets)
            else: #Mäter avstånd
                #Här

            #Räkna ut utbreddhet som ska användas sen antingen använda CP+bredd eller direkt area

                x_mat=[]
                y_mat=[]
                #def GOSPA_extended(x_mat, y_mat, p, c, alpha): #x_mat=truth CP,bredd y_mat=preds CP,bredd, p=1, c=cutoff dist, alpha=2
                    # Check input validity
                for target in current_targets:    
                    target=target.cpu()
                    #truth_cp=np.array([sum(target[0:len(target)-2:2])/4,sum(target[1:len(target)-2:2])/4])
                    truth_cp=np.array([target[0],target[1]])
                    #truth_size=np.array([math.sqrt(abs(truth_cp[0]-target[2])),math.sqrt(abs(truth_cp[1]-target[3])),math.sqrt(abs(truth_cp[0]-target[4])),math.sqrt(abs(truth_cp[1]-target[5]))])
                    truth_size=np.array([target[4],target[5]]) #fel dim?
                    x_mat.append([truth_cp,truth_size])
                for pred in alive_output:    
                    pred=pred.cpu()
                    #pred_cp=np.array([sum(pred[0:len(pred)-2:2])/4,sum(pred[1:len(pred)-2:2])/4])
                    pred_cp=np.array([pred[0],pred[1]])
                    #pred_size=np.array([math.sqrt(abs(pred_cp[0]-pred[2])),math.sqrt(abs(pred_cp[1]-pred[3])),math.sqrt(abs(pred_cp[0]-pred[4])),math.sqrt(abs(pred_cp[1]-pred[5]))])
                    pred_size=np.array([pred[4],pred[5]]) #fel dim?
                    y_mat.append([pred_cp,pred_size])

                c_mat_cp=np.zeros((len(y_mat),len(x_mat)))
                c_mat_size=np.zeros((len(y_mat),len(x_mat)))

                for i, pred in enumerate(y_mat):
                    for j, truth in enumerate(x_mat):
                        c_mat_cp[i][j]=np.linalg.norm(truth[0]-pred[0])
                        c_mat_size[i][j]=np.linalg.norm(truth[1][0]-pred[1][0])+np.linalg.norm(truth[1][1]-pred[1][1])

                dist=c_mat_cp+c_mat_size #Summerar size och cp, kan lägga till skalär på någon?
                #dist = torch.cdist(alive_output, current_targets, p=2) #Ställer upp distansmatris mellan sanna logits och objekt
                dist = torch.Tensor(dist).clamp_max(self.cutoff_distance) #Cuttar av vid maxdist
                c = torch.pow(input=dist, exponent=self.order)
                c = c.cpu()
                output_idx, target_idx = linear_sum_assignment(c) #Väljer ut och matchar optimal input med output
                indices.append((output_idx, target_idx))

            #Till hit
                for t, o in zip(output_idx, target_idx): #Kollar i distansmatrisen
                    loss_cost += c[t,o]
                    localization_cost += c[t,o].item()
                    #if c[t, o] < self.cutoff_distance:
                        #localization_cost += c[t, o].item()

                cardinality_error = abs(len(alive_output) - len(current_targets))
                if len(alive_output) > len(current_targets):
                    loss_cost += ((self.miss_cost/self.alpha) * cardinality_error)
                    false_target_cost += ((self.miss_cost/self.alpha) * cardinality_error)
                if len(alive_output) < len(current_targets):
                    loss_cost += ((self.miss_cost/self.alpha) * cardinality_error)
                    missed_target_cost += ((self.miss_cost/self.alpha) * cardinality_error)
                
            loss+=loss_cost
            loc+=localization_cost
            miss+=missed_target_cost
            false+=false_target_cost
            permutation+=permutation_length

        decomposition = {'localization': loc/bs, 'missed': miss/bs, 'false': false/bs,
                         'n_matched_objs': permutation/bs}
        
        loss=loss/bs

        return loss, indices, decomposition