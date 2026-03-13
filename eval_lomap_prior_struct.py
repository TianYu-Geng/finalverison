from absl import app, flags

import eval_prior_struct as prior_struct_eval


FLAGS = flags.FLAGS


def main(argv):
    FLAGS.use_lomap = True
    prior_struct_eval.main(argv)


if __name__ == "__main__":
    app.run(main)
