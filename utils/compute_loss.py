def smape(targets):
    """ This method validates SMAPE of the predicted values vs. the targets from the validation set

        :param targets - a list of target (true) values from the validation set

        :return calculated SMAPE or -1 in case the length of targets list does not equal to self._future_periods
    """
    smape = -1
    if len(targets) != self._future_periods:
        print("[ProphetModeller.smape] invalid target length: ", len(targets),
                ", expected length: ", self._future_periods)

    else:
        y_pred = self.get_forecast_only()['yhat']
        y_true = targets

        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        diff = np.abs(y_true - y_pred) / denominator
        diff[denominator == 0] = 0.0
        smape = np.mean(diff)

    return smape