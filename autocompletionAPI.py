import flask
import pandas as pd
from autocompletion.model import load_model, load_tokenizer, get_model_autocompletions, smi_tokenizer
from autocompletion.database_search import get_database_autocompletions

app = flask.Flask(__name__)


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    print('Autocomplete running', flush=True)
    SMILES = flask.request.args.get('SMILES').strip()
    if not SMILES:
        return flask.jsonify([])
    n_max_suggestions = int(flask.request.args.get('n_max_suggestions'))
    use_model = flask.request.args.get('use_model') == 'true'
    use_database = flask.request.args.get('use_database') == 'true'
    max_search_length = int(flask.request.args.get('max_search_length'))

    suggestions = []

    if use_model:
        suggestions.extend(
            get_model_autocompletions(SMILES, n_max_suggestions, app.model, app.tokenizer, max_depth=max_search_length))
    if use_database:
        suggestions.extend(get_database_autocompletions(SMILES, app.database_df, n=n_max_suggestions))

    sorted_suggestions = sorted(list(set(suggestions)), key=lambda x: len(smi_tokenizer(x)))
    sorted_suggestions = [x for x in sorted_suggestions if x != SMILES]
    return flask.jsonify(sorted_suggestions)

if __name__ == '__main__':
    app.model, app.tokenizer = load_model(), load_tokenizer('data/SMILES_Keyboard.csv')
    app.database_df = pd.read_csv('data/SMILES_Keyboard.csv')
    app.run()
