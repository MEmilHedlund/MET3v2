import torch
from modules.loss import MotLoss


def evaluate_gospa(data_generator, model, eval_params):
    with torch.no_grad():
        #model.eval()
        mot_loss = MotLoss(eval_params)
        gospa_total = 0
        gospa_loc = 0
        gospa_norm_loc = 0
        gospa_miss = 0
        gospa_false = 0

        for i in range(eval_params.n_samples):
            # Get batch from data generator and feed it to trained model
            batch, labels_pv, labels_bb, unique_ids, unique_label_ids, trajectories = data_generator.get_batch()
            #batch, labels, unique_ids, _, trajectories = data_generator.get_batch() #Kika så att det är rätt data som produceras?
            prediction, intermediate_predictions, encoder_prediction, aux_classifications, _ = model.forward(batch)

            # Compute GOSPA score
            prediction_in_format_for_loss = {'state': torch.cat((prediction.positions, prediction.velocities, prediction.bounding_boxes), dim=2),
                                             'logits': prediction.logits,
                                             'state_covariances': prediction.uncertainties ** 2}
            loss, _, decomposition = mot_loss.compute_orig_gospa_matching_EOT(prediction_in_format_for_loss, labels_pv, labels_bb, eval_params.loss.existence_prob_cutoff)
            gospa_total += loss.item()
            gospa_loc += decomposition['localization']
            gospa_norm_loc += decomposition['localization'] / decomposition['n_matched_objs'] if \
                decomposition['n_matched_objs'] != 0 else 0.0
            gospa_miss += decomposition['missed']
            gospa_false += decomposition['false']
            print(prediction.logits.sigmoid())

        model.train()
        gospa_total /= eval_params.n_samples
        gospa_loc /= eval_params.n_samples
        gospa_norm_loc /= eval_params.n_samples
        gospa_miss /= eval_params.n_samples
        gospa_false /= eval_params.n_samples
    return gospa_total, gospa_loc, gospa_norm_loc, gospa_miss, gospa_false
