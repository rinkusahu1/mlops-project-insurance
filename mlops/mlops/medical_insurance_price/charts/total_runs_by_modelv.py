from mlops.mi_util.analyticsm.data import load_data

# https://docs.mage.ai/visualizations/dashboards


@data_source
def data(*args, **kwargs):
    return load_data()
