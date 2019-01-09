"""
Custom stuff that is specific to the galaxy contest
"""

import numpy as np
import torch

class OptimisedDivGalaxyOutputLayer(object):
    """
    divisive normalisation, optimised for performance.
    """
    def __init__(self):
        self.question_slices = [slice(0, 3), slice(3, 5), slice(5, 7), slice(7, 9), slice(9, 13), slice(13, 15),
                                slice(15, 18), slice(18, 25), slice(25, 28), slice(28, 31), slice(31, 37)]

        self.normalisation_mask = self.generate_normalisation_mask

        # sequence of scaling steps to be undertaken.
        # First element is a slice indicating the values to be scaled. Second element is an index indicating the scale factor.
        # these have to happen IN ORDER else it doesn't work correctly.
        self.scaling_sequence = [
            (slice(3, 5), 1), # I: rescale Q2 by A1.2
            (slice(5, 13), 4), # II: rescale Q3, Q4, Q5 by A2.2
            (slice(15, 18), 0), # III: rescale Q7 by A1.1
            (slice(18, 25), 13), # IV: rescale Q8 by A6.1
            (slice(25, 28), 3), # V: rescale Q9 by A2.1
            (slice(28, 37), 7), # VI: rescale Q10, Q11 by A4.1
        ]

    @property
    def generate_normalisation_mask(self):
        """
        when the clipped input is multiplied by the normalisation mask, the normalisation denominators are generated.
        So then we can just divide the input by the normalisation constants (elementwise).
        """
        mask = np.zeros((37, 37), dtype='float32')
        for s in self.question_slices:
            mask[s, s] = 1.0
        return torch.Tensor(mask)

    def answer_probabilities(self, x):
        """
        normalise the answer groups for each question.
        """
        input_clipped = x

        normalisation_denoms = input_clipped.mm(self.normalisation_mask) + 1e-12 # small constant to prevent division by 0
        input_normalised = input_clipped / normalisation_denoms

        return input_normalised
        # return [input_normalised[:, s] for s in self.question_slices]

    def backable_normalize(self, x):
        # Should take input after answer_probability

        '''
        wts = torch.ones(x.shape)
        wts[:, 3:5] = x[:, 1].clone().reshape(-1, 1)
        wts[:, 5:13] = (x[:, 4].clone() * wts[:, 4]).reshape(-1, 1)
        wts[:, 15:18] = x[:, 0].clone().reshape(-1, 1)
        wts[:, 18:25] = x[:, 13].clone().reshape(-1, 1)
        wts[:, 25:28] = (x[:, 3].clone() * wts[:, 3]).reshape(-1, 1)
        wts[:, 28:37] = (x[:, 7].clone() * wts[:, 7]).reshape(-1, 1)
        return wts * x
        '''
        x35 = x[:, 3:5] * x[:, 1].reshape(-1, 1)
        x513 = x[:, 5:13] * (x35[:, 1] ).reshape(-1, 1)
        x1518 = x[:, 15:18] * (x[:, 0]).reshape(-1, 1)
        x1825 = x[:, 18:25] * (x[:, 13]).reshape(-1, 1)
        x2528 = x[:,25:28] * ( x35[:, 0]).reshape(-1, 1)
        x2837 = x[:, 28:37] * (x513[:, 2]).reshape(-1, 1)

        return torch.cat([x[:, :3], x35, x513, x[:, 13:15], x1518, x1825, x2528, x2837], dim = 1)
        

    def weighted_answer_probabilities(self, x):
        probs = self.answer_probabilities(x)

        # go through the rescaling sequence in order (6 steps)
        for probs_slice, scale_idx in self.scaling_sequence:
            probs[:, probs_slice] = probs[:, probs_slice] * probs[:, scale_idx].reshape(-1, 1)

        return probs

    def predictions(self, x):
        return self.weighted_answer_probabilities(x)

    def predictions_no_normalisation(self, x):
        """
        Predict without normalisation. This can be used for the first few chunks to find good parameters.
        """
        input = x
        input_clipped = x.clip(input, 0, 1) # clip on both sides here, any predictions over 1.0 are going to get normalised away anyway.
        return input_clipped

    def error(self, x, normalisation=True):
        if normalisation:
            predictions = self.predictions(x)
        else:
            predictions = self.predictions_no_normalisation(x)
        error = ((predictions - target) ** 2).mean()
        return error

