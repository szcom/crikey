import numpy as np
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from scipy.io import wavfile
import os
import sys
from kdllib import overlap
from kdllib import load_checkpoint, theano_one_hot, concatenate
from kdllib import fetch_fruitspeech, list_iterator
from kdllib import np_zeros, GRU, GRUFork, dense_to_one_hot
from kdllib import make_weights, make_biases, relu, run_loop
from kdllib import as_shared, adam, gradient_clipping
from kdllib import get_values_from_function, set_shared_variables_in_function
from kdllib import soundsc, categorical_crossentropy
from kdllib import relu, softmax



if __name__ == "__main__":
    import argparse

    speech = fetch_fruitspeech()
    X = speech["data"]
    fs = speech["sample_rate"]
    X = np.array([x.astype(theano.config.floatX) for x in X])
    # reslice to 2D in sets of 128, 4
    def _s(x):
        xi = x[:len(x) - len(x) % 4]
        # chunk it
        xi = xi.reshape((-1, 4))
        #xi = overlap(xi, 4, 1)
        return xi

    X_windows = []
    for x in X:
        xw = _s(x)
        X_windows.append(xw)
    X = X_windows

    y = [Xi[1:, 0][:, None] for Xi in X]
    X = [Xi[:-1] for Xi in X]

    minibatch_size = 1
    n_epochs = 5000  # Used way at the bottom in the training loop!
    checkpoint_every_n = 10
    random_state = np.random.RandomState(1999)

    train_itr = list_iterator([X, y], minibatch_size, axis=1,
                              stop_index=1, randomize=True, make_mask=True)
    valid_itr = list_iterator([X, y], minibatch_size, axis=1,
                              start_index=105 - minibatch_size,
                              randomize=True, make_mask=True)
    X_mb, X_mb_mask, y_mb, y_mb_mask = next(train_itr)
    train_itr.reset()
    n_hid = 512
    n_frame = 4
    n_proj = n_frame * n_hid
    n_bins = 256
    input_dim = X_mb.shape[-1]
    n_hid_mlp = 1024
    n_pred_proj = n_bins
    cut_len = 60

    desc = "Speech generation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', '--sample',
                        help='Sample from a checkpoint file',
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
    if args.sample is not None:
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
    X_sym.tag.test_value = X_mb[:cut_len]
    X_mask_sym = tensor.matrix("X_mask_sym")
    X_mask_sym.tag.test_value = X_mb_mask[:cut_len]
    y_sym = tensor.tensor3("y_sym")
    y_sym.tag.test_value = y_mb[:cut_len]
    y_mask_sym = tensor.matrix("y_mask_sym")
    y_mask_sym.tag.test_value = y_mb_mask[:cut_len]

    init_h1 = tensor.matrix("init_h1")
    init_h1.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h2 = tensor.matrix("init_h2")
    init_h2.tag.test_value = np_zeros((minibatch_size, n_hid))

    init_h3 = tensor.matrix("init_h3")
    init_h3.tag.test_value = np_zeros((minibatch_size, n_hid))

    params = []
    biases = []

    cell1 = GRU(input_dim, n_hid, random_state)
    cell2 = GRU(n_hid, n_hid, random_state)
    cell3 = GRU(n_hid, n_hid, random_state)

    params += cell1.get_params()
    params += cell2.get_params()
    params += cell3.get_params()

    inp_to_h1 = GRUFork(input_dim, n_hid, random_state)
    inp_to_h2 = GRUFork(input_dim, n_hid, random_state)
    inp_to_h3 = GRUFork(input_dim, n_hid, random_state)
    h1_to_h2 = GRUFork(n_hid, n_hid, random_state)
    h1_to_h3 = GRUFork(n_hid, n_hid, random_state)
    h2_to_h3 = GRUFork(n_hid, n_hid, random_state)

    params += inp_to_h1.get_params()
    params += inp_to_h2.get_params()
    params += inp_to_h3.get_params()
    params += h1_to_h2.get_params()
    params += h1_to_h3.get_params()
    params += h2_to_h3.get_params()

    biases += inp_to_h1.get_biases()
    biases += inp_to_h2.get_biases()
    biases += inp_to_h3.get_biases()
    biases += h1_to_h2.get_biases()
    biases += h1_to_h3.get_biases()
    biases += h2_to_h3.get_biases()

    h1_to_outs, = make_weights(n_hid, [n_proj], random_state)
    h2_to_outs, = make_weights(n_hid, [n_proj], random_state)
    h3_to_outs, = make_weights(n_hid, [n_proj], random_state)
    b_to_outs, = make_biases([n_proj])

    params += [h1_to_outs, h2_to_outs, h3_to_outs]
    biases += [b_to_outs]

    mlp1_w, = make_weights(n_proj + 2 * n_frame, [n_hid_mlp], random_state)
    mlp2_w, mlp3_w = make_weights(n_hid_mlp, 2 * [n_hid_mlp], random_state)
    pred_w, = make_weights(n_hid_mlp, [n_bins], random_state)

    mlp1_b, mlp2_b, mlp3_b = make_biases(3 * [n_hid_mlp])
    pred_b, = make_biases([n_bins])

    params += [mlp1_w, mlp1_b]
    params += [mlp2_w, mlp2_b]
    params += [mlp3_w, mlp3_b]
    params += [pred_w, pred_b]
    biases += [mlp1_b, mlp2_b, mlp3_b, pred_b]

    # Done initializing

    inpt = X_sym
    target = y_sym
    mask = X_mask_sym

    inp_h1, inpgate_h1 = inp_to_h1.proj(inpt)
    inp_h2, inpgate_h2 = inp_to_h2.proj(inpt)
    inp_h3, inpgate_h3 = inp_to_h3.proj(inpt)

    def step(xinp_h1_t, xgate_h1_t,
             xinp_h2_t, xgate_h2_t,
             xinp_h3_t, xgate_h3_t,
             h1_tm1, h2_tm1, h3_tm1):
        h1_t = cell1.step(xinp_h1_t, xgate_h1_t,
                          h1_tm1)
        h1inp_h2, h1gate_h2 = h1_to_h2.proj(h1_t)
        h1inp_h3, h1gate_h3 = h1_to_h3.proj(h1_t)

        h2_t = cell2.step(xinp_h2_t + h1inp_h2,
                          xgate_h2_t + h1gate_h2, h2_tm1)

        h2inp_h3, h2gate_h3 = h2_to_h3.proj(h2_t)

        h3_t = cell3.step(xinp_h3_t + h1inp_h3 + h2inp_h3,
                          xgate_h3_t + h1gate_h3 + h2gate_h3,
                          h3_tm1)
        return h1_t, h2_t, h3_t


    init_x = tensor.fmatrix()
    init_x.tag.test_value = np_zeros((minibatch_size, input_dim)).astype(theano.config.floatX)
    srng = RandomStreams(1999)

    def sample_step(x_tm1, h1_tm1, h2_tm1, h3_tm1, prev_tm1):
        xinp_h1_t, xgate_h1_t = inp_to_h1.proj(x_tm1)
        xinp_h2_t, xgate_h2_t = inp_to_h2.proj(x_tm1)
        xinp_h3_t, xgate_h3_t = inp_to_h3.proj(x_tm1)

        h1_t = cell1.step(xinp_h1_t, xgate_h1_t, h1_tm1)
        h1inp_h2, h1gate_h2 = h1_to_h2.proj(h1_t)
        h1inp_h3, h1gate_h3 = h1_to_h3.proj(h1_t)


        h2_t = cell2.step(xinp_h2_t + h1inp_h2,
                          xgate_h2_t + h1gate_h2, h2_tm1)

        h2inp_h3, h2gate_h3 = h2_to_h3.proj(h2_t)

        h3_t = cell3.step(xinp_h3_t + h1inp_h3 + h2inp_h3,
                          xgate_h3_t + h1gate_h3 + h2gate_h3,
                          h3_tm1)
        out_t = h1_t.dot(h1_to_outs) + h2_t.dot(h2_to_outs) + h3_t.dot(
            h3_to_outs) + b_to_outs
        theano.printing.Print("prev_tm1.shape")(prev_tm1.shape)
        theano.printing.Print("x_tm1.shape")(x_tm1.shape)
        theano.printing.Print("out_t.shape")(out_t.shape)
        full_out_t = tensor.concatenate((prev_tm1, x_tm1, out_t), axis=-1)
        mlp1_t = relu(full_out_t.dot(mlp1_w) + mlp1_b)
        mlp2_t = relu(mlp1_t.dot(mlp2_w) + mlp2_b)
        mlp3_t = relu(mlp2_t.dot(mlp3_w) + mlp3_b)
        pred_t = softmax(mlp3_t.dot(pred_w) + pred_b)
        x_t = pred_t
        return x_t, h1_t, h2_t, h3_t

    init_prev = tensor.fmatrix()
    init_prev.tag.test_value = np.zeros_like(X_mb[0])

    (sampled, h1_s, h2_s, h3_s) = sample_step(
        init_x, init_h1, init_h2, init_h3, init_prev)
    theano.printing.Print("sampled.shape")(sampled.shape)

    (h1, h2, h3), updates = theano.scan(
        fn=step,
        sequences=[inp_h1, inpgate_h1,
                   inp_h2, inpgate_h2,
                   inp_h3, inpgate_h3],
        outputs_info=[init_h1, init_h2, init_h3])

    outs = h1.dot(h1_to_outs) + h2.dot(h2_to_outs) + h3.dot(h3_to_outs) + b_to_outs
    outs_shape = outs.shape
    first_prev = tensor.zeros_like(inpt[0]).dimshuffle('x', 0, 1)
    prev_n = tensor.concatenate((first_prev, inpt[:-1]), axis=0)
    """
    for i in range(4):
        partial_outs = outs[:, :, i * n_hid: (i + 1) * n_hid]
        theano.printing.Print("partial_outs.shape")(partial_outs.shape)
        theano.printing.Print("prev_n.shape")(prev_n.shape)
        theano.printing.Print("inpt.shape")(inpt.shape)
    raise ValueError()
    """
    theano.printing.Print("target.shape")(target.shape)
    full_outs = tensor.concatenate((prev_n, inpt, outs), axis=-1)
    theano.printing.Print("full_outs.shape")(full_outs.shape)
    shp = full_outs.shape
    mlp_inpt = full_outs.reshape((-1, shp[-1]))
    mlp1 = relu(mlp_inpt.dot(mlp1_w) + mlp1_b)
    mlp2 = relu(mlp1.dot(mlp2_w) + mlp2_b)
    mlp3 = relu(mlp2.dot(mlp3_w) + mlp3_b)
    pred = softmax(mlp3.dot(pred_w) + pred_b)
    theano.printing.Print("pred.shape")(pred.shape)
    pred = pred.reshape((shp[0], shp[1], -1))
    theano.printing.Print("pred.shape")(pred.shape)
    target = target.reshape((shp[0], shp[1]))  # target has dim 1
    target = theano_one_hot(target, n_classes=n_bins)
    theano.printing.Print("target.shape")(target.shape)
    cost = categorical_crossentropy(pred, target)
    theano.printing.Print("mask.shape")(mask.shape)
    theano.printing.Print("cost.shape")(cost.shape)

    cost = cost * mask.dimshuffle(0, 1)
    # sum over sequence length and features, mean over minibatch
    cost = cost.sum(axis=0).mean()

    l2_penalty = 0
    for p in list(set(params) - set(biases)):
        l2_penalty += (p ** 2).sum()

    cost = cost + 1E-3 * l2_penalty
    grads = tensor.grad(cost, params)
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
        sample_function = checkpoint_dict["sample_function"]
        """
        trained_weights = get_values_from_function(
            saved_checkpoint["train_function"])
        set_shared_variables_in_function(train_function, trained_weights)
        """
    else:
        train_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym,
                                          init_h1, init_h2, init_h3],
                                         [cost, h1, h2, h3],
                                         updates=updates,
                                         on_unused_input="warn")
        cost_function = theano.function([X_sym, X_mask_sym, y_sym, y_mask_sym,
                                          init_h1, init_h2, init_h3],
                                         [cost, h1, h2, h3],
                                         on_unused_input="warn")
        predict_function = theano.function([X_sym, X_mask_sym,
                                          init_h1, init_h2, init_h3],
                                         [pred, h1, h2, h3],
                                        on_unused_input="warn")
        sample_function = theano.function([init_x, init_h1, init_h2,
                                           init_h3, init_prev],
                                          [sampled, h1_s, h2_s, h3_s],
                                          on_unused_input="warn")
        print("Beginning training loop")
        checkpoint_dict = {}
        checkpoint_dict["train_function"] = train_function
        checkpoint_dict["cost_function"] = cost_function
        checkpoint_dict["predict_function"] = predict_function
        checkpoint_dict["sample_function"] = sample_function


    def _loop(function, itr):
        prev_h1, prev_h2, prev_h3 = [np_zeros((minibatch_size, n_hid))
                                     for i in range(3)]
        X_mb, X_mb_mask, y_mb, y_mb_mask = next(itr)
        n_cuts = len(X_mb) // cut_len + 1
        partial_costs = []
        for n in range(n_cuts):
            start = n * cut_len
            stop = (n + 1) * cut_len
            if len(X_mb[start:stop]) < cut_len:
                # skip end edge case
                break
                new_len = cut_len - len(X_mb) % cut_len
                zeros = np.zeros((new_len, X_mb.shape[1],
                                  X_mb.shape[2]))
                zeros = zeros.astype(X_mb.dtype)
                mask_zeros = np.zeros((new_len, X_mb_mask.shape[1]))
                mask_zeros = mask_zeros.astype(X_mb_mask.dtype)
                X_mb = np.concatenate((X_mb, zeros), axis=0)
                X_mb_mask = np.concatenate((X_mb_mask, mask_zeros), axis=0)
                assert len(X_mb[start:stop]) == cut_len
                assert len(X_mb_mask[start:stop]) == cut_len
            rval = function(X_mb[start:stop],
                            X_mb_mask[start:stop],
                            y_mb[start:stop], y_mb_mask[start:stop],
                            prev_h1, prev_h2, prev_h3)
            current_cost = rval[0]
            prev_h1, prev_h2, prev_h3 = rval[1:4]
            prev_h1 = prev_h1[-1]
            prev_h2 = prev_h2[-1]
            prev_h3 = prev_h3[-1]
        partial_costs.append(current_cost)
        return partial_costs

run_loop(_loop, train_function, train_itr, cost_function, valid_itr,
         n_epochs=n_epochs, checkpoint_dict=checkpoint_dict,
         checkpoint_every_n=checkpoint_every_n, skip_minimums=True)
