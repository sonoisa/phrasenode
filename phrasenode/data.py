import os
from gtd.io import Workspace

# Set location of local data directory from environment variable
env_var = 'WEBREP_DATA'
if env_var not in os.environ:
    assert False, env_var + ' environmental variable must be set.'
root = os.environ[env_var]

output_var = 'OUTPUT_DATA'
if output_var not in os.environ:
    assert False, output_var + ' environmental variable must be set.'
output_root = os.environ[output_var]

# define workspace
workspace = Workspace(root)
workspace.add_dir("vocab", "vocab")
workspace.add_dir("phrase_node", "phrase-node-dataset")
workspace.add_dir("word_embeddings", "word_embeddings")

output_workspace = Workspace(output_root)
output_workspace.add_dir("experiments", "experiments")
