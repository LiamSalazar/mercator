.PHONY: data features train backtest report
data:
python -m mercator_data.ingest
features:
python -m mercator_nlp.build_features
train:
python -m mercator_states.train && python -m mercator_policy.train
backtest:
python -m mercator_risk.backtest
report:
python -m mercator_utils.report
