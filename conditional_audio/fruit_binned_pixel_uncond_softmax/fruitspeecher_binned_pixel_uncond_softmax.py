import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import load_checkpoint, conv2d, conv2d_transpose
from kdllib import make_weights, make_biases
from kdllib import make_conv_weights
from kdllib import fetch_fruitspeech_spectrogram, list_iterator
from kdllib import theano_one_hot
from kdllib import softmax, tanh
from kdllib import adam, gradient_clipping, categorical_crossentropy
from kdllib import run_loop


if __name__ == "__main__":
    import argparse

    speech = fetch_fruitspeech_spectrogram()
    X = speech["data"]
    y = speech["target"]
    vocabulary = speech["vocabulary"]
    vocabulary_size = speech["vocabulary_size"]
    reconstruct = speech["reconstruct"]
    fs = speech["sample_rate"]
    X = np.array([x.astype(theano.config.floatX) for x in X])
    y = np.array([yy.astype(theano.config.floatX) for yy in y])

    minibatch_size = 1
    n_epochs = 200  # Used way at the bottom in the training loop!
    checkpoint_every_n = 10
    n_bins = 10
    random_state = np.random.RandomState(1999)

    """
    train_itr = list_iterator([X, y], minibatch_size, axis=1,
                              stop_index=105, randomize=True, make_mask=True)
    valid_itr = list_iterator([X, y], minibatch_size, axis=1,
                              start_index=105 - minibatch_size,
                              randomize=True, make_mask=True)
    """
    train_itr = list_iterator([X], minibatch_size, axis=1,
                              stop_index=minibatch_size, randomize=True,
                              make_mask=True)
    valid_itr = list_iterator([X], minibatch_size, axis=1,
                              start_index=105 - minibatch_size,
                              randomize=True, make_mask=True)
    X_mb, X_mb_mask = next(train_itr)
    train_itr.reset()

    desc = "Speech generation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--sample',
                        help='Sample from a checkpoint file',
                        default=None,
                        required=False)
    parser.add_argument('-p', '--plot',
                        help='Plot training curves from a checkpoint file',
                        default=None,
                        required=False)
    parser.add_argument('-w', '--write',
                        help='The string to write out (default first minibatch)',
                        default=None,
                        required=False)

    def restricted_int(x):
        if x is None:
            # None makes it "auto" sample
            return x
        x = int(x)
        if x < 1:
            raise argparse.ArgumentTypeError("%r not range [1, inf]" % (x,))
        return x
    parser.add_argument('-sl', '--sample_length',
                        help='Number of steps to sample, default is automatic',
                        type=restricted_int,
                        default=None,
                        required=False)
    parser.add_argument('-c', '--continue', dest="cont",
                        help='Continue training from another saved model',
                        default=None,
                        required=False)
    args = parser.parse_args()
    if args.plot is not None or args.sample is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        if args.sample is not None:
            checkpoint_file = args.sample
        else:
            checkpoint_file = args.plot
        if not os.path.exists(checkpoint_file):
            raise ValueError("Checkpoint file path %s" % checkpoint_file,
                             " does not exist!")
        print(checkpoint_file)
        checkpoint_dict = load_checkpoint(checkpoint_file)
        train_costs = checkpoint_dict["train_costs"]
        valid_costs = checkpoint_dict["valid_costs"]
        plt.plot(train_costs)
        plt.plot(valid_costs)
        plt.savefig("costs.png")

        X_mb, X_mb_mask, c_mb, c_mb_mask = next(valid_itr)
        valid_itr.reset()
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]
        prev_kappa = np_zeros((minibatch_size, att_size))
        prev_w = np_zeros((minibatch_size, n_chars))
        if args.sample is not None:
            predict_function = checkpoint_dict["predict_function"]
            attention_function = checkpoint_dict["attention_function"]
            sample_function = checkpoint_dict["sample_function"]
            if args.write is not None:
                sample_string = args.write
                print("Sampling using sample string %s" % sample_string)
                oh = dense_to_one_hot(
                    np.array([vocabulary[c] for c in sample_string]),
                    vocabulary_size)
                c_mb = np.zeros(
                    (len(oh), minibatch_size, oh.shape[-1])).astype(c_mb.dtype)
                c_mb[:len(oh), :, :] = oh[:, None, :]
                c_mb = c_mb[:len(oh)]
                c_mb_mask = np.ones_like(c_mb[:, :, 0])

            if args.sample_length is None:
                raise ValueError("NYI - use -sl or --sample_length ")
            else:
                fixed_steps = args.sample_length
                completed = []
                init_x = np.zeros_like(X_mb[0])
                for i in range(fixed_steps):
                    rvals = sample_function(init_x, c_mb, c_mb_mask, prev_h1, prev_h2,
                                            prev_h3, prev_kappa, prev_w)
                    sampled, h1_s, h2_s, h3_s, k_s, w_s, stop_s, stop_h = rvals
                    completed.append(sampled)
                    # cheating sampling...
                    #init_x = X_mb[i]
                    init_x = sampled
                    prev_h1 = h1_s
                    prev_h2 = h2_s
                    prev_h3 = h3_s
                    prev_kappa = k_s
                    prev_w = w_s
                cond = c_mb
                print("Completed sampling after %i steps" % fixed_steps)
            completed = np.array(completed).transpose(1, 0, 2)
            rlookup = {v: k for k, v in vocabulary.items()}
            all_strings = []
            for yi in y:
                ex_str = "".join([rlookup[c]
                                  for c in np.argmax(yi, axis=1)])
                all_strings.append(ex_str)
            for i in range(len(completed)):
                ex = completed[i]
                ex_str = "".join([rlookup[c]
                                  for c in np.argmax(cond[:, i], axis=1)])
                s = "gen_%s_%i.wav" % (ex_str, i)
                ii = reconstruct(ex)
                wavfile.write(s, fs, soundsc(ii))
                if ex_str in all_strings:
                    inds = [n for n, s in enumerate(all_strings)
                            if ex_str == s]
                    ind = inds[0]
                    it = reconstruct(X[ind])
                    s = "orig_%s_%i.wav" % (ex_str, i)
                    wavfile.write(s, fs, soundsc(it))
        valid_itr.reset()
        print("Sampling complete, exiting...")
        sys.exit()
    else:
        print("No plotting arguments, starting training mode!")

    X_sym = tensor.tensor3("X_sym")
    X_sym.tag.test_value = X_mb
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask


    params = []
    biases = []

    n_conv1 = 128
    k_conv1 = (1, 3)
    k_conv1_hid = (1, 3)

    conv1_w, = make_conv_weights(1, [n_conv1,], k_conv1, random_state)
    conv1_b, = make_biases([n_conv1,])
    params += [conv1_w, conv1_b]
    biases += [conv1_b]

    # Might become 3* for GRU or 4* for LSTM
    conv1_hid, = make_conv_weights(n_conv1, [n_conv1,], k_conv1_hid, random_state)
    params += [conv1_hid]

    tconv1_w, = make_conv_weights(n_conv1, [n_bins,], k_conv1, random_state)
    tconv1_b, = make_biases([n_bins,])
    params += [tconv1_w, tconv1_b]
    biases += [tconv1_b]

    theano.printing.Print("X_sym.shape")(X_sym.shape)
    # add channel dim
    im = X_sym.dimshuffle(1, 'x', 0, 2)
    target = im
    shp = im.shape
    # careful shift to avoid leakage
    conv1 = conv2d(im, conv1_w, conv1_b, border_mode=(0, k_conv1[1] + 1))
    theano.printing.Print("conv1.shape")(conv1.shape)
    conv1 = conv1[:, :, :, :shp[3]]
    theano.printing.Print("conv1.shape")(conv1.shape)
    r_conv1 = conv1.dimshuffle(2, 1, 0, 3)
    theano.printing.Print("r_conv1.shape")(r_conv1.shape)
    shp = r_conv1.shape

    init_hidden = tensor.zeros((minibatch_size, n_conv1, 1, shp[3]),
                                dtype=theano.config.floatX)
    # weirdness in broadcast
    if minibatch_size == 1:
        init_hidden = tensor.unbroadcast(init_hidden, 0, 2)
    else:
        init_hidden = tensor.unbroadcast(init_hidden, 2)
    theano.printing.Print("init_hidden.shape")(init_hidden.shape)
    # recurrent function (using tanh activation function)
    def step(in_t, h_tm1):
        theano.printing.Print("in_t.shape")(in_t.shape)
        theano.printing.Print("h_tm1.shape")(h_tm1.shape)
        h_i = conv2d(h_tm1, conv1_hid, border_mode="half")
        theano.printing.Print("h_i.shape")(h_i.shape)
        in_i = in_t.dimshuffle(1, 0, 'x', 2)
        theano.printing.Print("in_i.shape")(in_i.shape)
        h_t = tanh(in_i + h_i)
        # need to add broadcast dims back to keep scan happy
        theano.printing.Print("h_t.shape")(h_t.shape)
        return h_t
    h, updates = theano.scan(fn=step,
                             sequences=[r_conv1],
                             outputs_info=[init_hidden])
    h = tensor.unbroadcast(h, 0, 1, 2, 3, 4)
    # remove spurious axis
    h = h[:, :, :, 0]
    # dimshuffle back to bc01
    theano.printing.Print("h.shape")(h.shape)
    h = h.dimshuffle(1, 2, 0, 3)
    theano.printing.Print("h.shape")(h.shape)
    pred = conv2d_transpose(h, tconv1_w, tconv1_b, border_mode="half")
    pred = softmax(pred)
    # transpose to put one hot last
    theano.printing.Print("pred.shape")(pred.shape)
    pred = pred.dimshuffle(0, 2, 3, 1)
    theano.printing.Print("pred.shape")(pred.shape)
    theano.printing.Print("target.shape")(target.shape)
    target = theano_one_hot(target, n_classes=n_bins)
    # remove spurious channel
    target = target[:, 0]
    theano.printing.Print("target.shape")(target.shape)
    cost = categorical_crossentropy(pred, target)
    cost = cost.dimshuffle(1, 2, 0)
    theano.printing.Print("cost.shape")(cost.shape)
    shp = cost.shape
    cost = cost.reshape((shp[0] * shp[1], -1))
    theano.printing.Print("cost.shape")(cost.shape)
    cost = cost.sum(axis=0).mean()
    theano.printing.Print("cost.shape")(cost.shape)

    l2_penalty = 0
    for p in list(set(params) - set(biases)):
        l2_penalty += (p ** 2).sum()

    reg_cost = cost + 1E-3 * l2_penalty

    grads = tensor.grad(reg_cost, params)
    grads = gradient_clipping(grads, 10.)

    learning_rate = 1E-4

    opt = adam(params, learning_rate)
    updates = opt.updates(params, grads)

    if args.cont is not None:
        print("Continuing training from saved model")
        continue_path = args.cont
        if not os.path.exists(continue_path):
            raise ValueError("Continue model %s, path not "
                             "found" % continue_path)
        saved_checkpoint = load_checkpoint(continue_path)
        checkpoint_dict = saved_checkpoint
        train_function = checkpoint_dict["train_function"]
        cost_function = checkpoint_dict["cost_function"]
        predict_function = checkpoint_dict["predict_function"]
    else:
        train_function = theano.function([X_sym, X_mask_sym],
                                         [cost],
                                         updates=updates,
                                         on_unused_input='warn')
        cost_function = theano.function([X_sym, X_mask_sym],
                                        [cost],
                                        on_unused_input='warn')
        predict_function = theano.function([X_sym, X_mask_sym],
                                           [pred],
                                           on_unused_input='warn')
        print("Beginning training loop")
        checkpoint_dict = {}
        checkpoint_dict["train_function"] = train_function
        checkpoint_dict["cost_function"] = cost_function
        checkpoint_dict["predict_function"] = predict_function


    def _loop(function, itr):
        X_mb, X_mb_mask = next(itr)
        rval = function(X_mb[:10, :, :20], X_mb_mask)
        cost = rval[0]
        return [cost]

run_loop(_loop, train_function, train_itr, cost_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n=checkpoint_every_n, skip_minimums=True)
