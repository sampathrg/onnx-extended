"""
.. _l-example-bench-gemm:

Measuring performance about Gemm
================================

Differents types, differents backend, differents

Onnx Model
++++++++++
"""
import pprint
import platform
from itertools import product
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame, pivot_table
from onnx import TensorProto
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_opsetid,
)
from onnx.checker import check_model
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession, SessionOptions, get_available_providers
from onnxruntime.capi._pybind_state import (
    OrtValue as C_OrtValue,
    OrtDevice as C_OrtDevice,
)
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    NotImplemented,
    InvalidGraph,
    InvalidArgument,
)

try:
    from onnx_extended.reference import CReferenceEvaluator
except ImportError:
    CReferenceEvaluator = ReferenceEvaluator
from onnx_extended.ext_test_case import unit_test_going, measure_time

try:
    from onnx_extended.validation.cuda.cuda_example_py import get_device_prop
    from onnx_extended.ortops.tutorial.cuda import get_ort_ext_libs
except ImportError:

    def get_device_prop():
        return {"name": "CPU"}

    def get_ort_ext_libs():
        return None


properties = get_device_prop()
pprint.pprint(properties)


###################################
# Model to benchmark
# ++++++++++++++++++


def create_model(
    mat_type=TensorProto.FLOAT, provider="CUDAExecutionProvider", domain="com.microsoft"
):
    A = make_tensor_value_info("A", mat_type, [None, None])
    B = make_tensor_value_info("B", mat_type, [None, None])
    outputs = [make_tensor_value_info("C", mat_type, [None, None])]
    inits = []
    if domain != "":
        if provider != "CUDAExecutionProvider":
            return None
        f8 = False
        if domain == "com.microsoft":
            op_name = "GemmFloat8"
            computeType = "CUBLAS_COMPUTE_32F_FAST_TF32"
            node_output = ["C"]
        elif mat_type == TensorProto.FLOAT:
            op_name = "CustomGemmFloat"
            computeType = "CUBLAS_COMPUTE_32F_FAST_TF32"
            node_output = ["C", "time"]
            outputs.append(make_tensor_value_info("time", TensorProto.DOUBLE, [None]))
        elif mat_type == TensorProto.FLOAT16:
            op_name = "CustomGemmFloat16"
            computeType = "CUBLAS_COMPUTE_16F"
            node_output = ["C", "time"]
            outputs.append(make_tensor_value_info("time", TensorProto.DOUBLE, [None]))
        elif mat_type in (TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2):
            f8 = True
            op_name = "CustomGemmFloat8E4M3FN"
            computeType = "CUBLAS_COMPUTE_32F_FAST_TF32"
            node_output = ["C", "time"]
            outputs = [
                make_tensor_value_info("C", TensorProto.FLOAT16, [None, None]),
                make_tensor_value_info("time", TensorProto.DOUBLE, [None]),
            ]
            inits.append(from_array(numpy.array([1], dtype=numpy.float32), name="I"))
        else:
            return None
        node_kw = dict(
            alpha=1.0,
            transA=1,
            domain=domain,
            computeType=computeType,
            fastAccumulationMode=1,
            rowMajor=0 if op_name == "CustomGemmFloat8E4M3FN" else 1,
        )
        node_kw["name"] = (
            f"{mat_type}.{len(node_output)}.{len(outputs)}."
            f"{domain}..{node_kw['rowMajor']}.."
            f"{node_kw['fastAccumulationMode']}..{node_kw['computeType']}.."
            f"{f8}"
        )
        nodes = [
            make_node(
                op_name,
                ["A", "B", "I", "I", "I"] if f8 else ["A", "B"],
                node_output,
                **node_kw,
            ),
        ]
    else:
        nodes = [
            make_node("Gemm", ["A", "B"], ["C"], transA=1, beta=0.0),
        ]
    graph = make_graph(nodes, "a", [A, B], outputs, inits)
    if mat_type < 16:
        # regular type
        opset, ir = 18, 8
    else:
        opset, ir = 19, 9
    onnx_model = make_model(
        graph,
        opset_imports=[
            make_opsetid("", opset),
            make_opsetid("com.microsoft", 1),
            make_opsetid("onnx_extented.ortops.tutorial.cuda", 1),
        ],
        ir_version=ir,
    )
    check_model(onnx_model)
    return onnx_model


create_model()

###########################################
# A model to cast


def create_cast(to, cuda=False):
    A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
    C = make_tensor_value_info("C", to, [None, None])
    if cuda:
        nodes = [
            make_node("Cast", ["A"], ["Cc"], to=to),
            make_node("MemcpyFromHost", ["Cc"], ["C"]),
        ]
    else:
        nodes = [make_node("Cast", ["A"], ["C"], to=to)]
    graph = make_graph(nodes, "a", [A], [C])
    if to < 16:
        # regular type
        opset, ir = 18, 8
    else:
        opset, ir = 19, 9
    onnx_model = make_model(
        graph, opset_imports=[make_opsetid("", opset)], ir_version=ir
    )
    if not cuda:
        # OpType: MemcpyFromHost
        check_model(onnx_model)
    return onnx_model


create_cast(TensorProto.FLOAT16)


##############################
# Performance
# +++++++++++
#
# The benchmark will run the following configurations.

types = [
    TensorProto.FLOAT8E4M3FN,
    TensorProto.FLOAT,
    TensorProto.FLOAT16,
    # TensorProto.BFLOAT16,
    # TensorProto.UINT32,
    # TensorProto.INT32,
    # TensorProto.INT16,
    # TensorProto.INT8,
]
engine = [InferenceSession, CReferenceEvaluator]
providers = [
    ["CUDAExecutionProvider", "CPUExecutionProvider"],
    ["CPUExecutionProvider"],
]
# M, N, K
# we use multiple of 8, otherwise, float8 does not work.
dims = [
    (32, 32, 32),
    (56, 64, 72),
    (64, 64, 64),
    (64, 72, 80),
    (128, 128, 128),
    (256, 256, 256),
    (400, 400, 400),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    # (16384, 16384, 16384),
]

domains = ["onnx_extented.ortops.tutorial.cuda", "", "com.microsoft"]


####################################
# Let's cache the matrices involved.


def to_ort_value(m):
    device = C_OrtDevice(C_OrtDevice.cpu(), C_OrtDevice.default_memory(), 0)
    ort_value = C_OrtValue.ortvalue_from_numpy(m, device)
    return ort_value


matrices = {}
matrices_cuda = {}
for m, n, k in dims:
    for tt in types:
        for i, j in [(m, k), (k, n), (k, m)]:
            if (tt, i, j) in matrices:
                continue
            # CPU
            try:
                sess = InferenceSession(
                    create_cast(tt).SerializeToString(),
                    providers=["CPUExecutionProvider"],
                )
                cpu = True
            except (InvalidGraph, InvalidArgument, NotImplemented):
                # not support by this version of onnxruntime
                cpu = False

            if cpu:
                vect = (numpy.random.randn(i, j) * 10).astype(numpy.float32)
                ov = to_ort_value(vect)
                ovtt = sess._sess.run_with_ort_values({"A": ov}, ["C"], None)[0]
                matrices[tt, i, j] = ovtt
            else:
                continue

            # CUDA
            if "CUDAExecutionProvider" not in get_available_providers():
                # No CUDA
                continue
            sess = InferenceSession(
                create_cast(tt, cuda=True).SerializeToString(),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            vect = (numpy.random.randn(i, j) * 10).astype(numpy.float32)
            ov = to_ort_value(vect)
            ovtt = sess._sess.run_with_ort_values({"A": ov}, ["C"], None)[0]
            matrices_cuda[tt, i, j] = ovtt

print(f"{len(matrices)} matrices were created.")

###################################
# Let's run the benchmark


data = []
errors = []
pbar = tqdm(list(product(types, engine, providers, dims, domains)))
for tt, engine, provider, dim, domain in pbar:
    if (
        tt in {TensorProto.FLOAT8E4M3FN, TensorProto.FLOAT8E5M2}
        and properties.get("major", 0) < 9
    ):
        # f8 not available
        if provider[0] == "CPUExecutionProvider":
            continue
        errors.append(
            f"f8 not available, major={properties.get('major', 0)}, "
            f"tt={tt}, provider={provider!r}, domain={domain!r}."
        )
        continue
    elif provider[0] == "CPUExecutionProvider" and max(dim) > 2000:
        # too long
        continue
    if max(dim) <= 200:
        repeat, number = 50, 25
    elif max(dim) <= 256:
        repeat, number = 25, 10
    else:
        repeat, number = 10, 4

    onx = create_model(tt, provider=provider[0], domain=domain)
    if onx is None:
        if provider[0] == "CPUExecutionProvider":
            continue
        errors.append(
            f"No model for tt={tt}, provider={provider!r}, domain={domain!r}."
        )
        continue
    with open(f"plot_bench_gemm_{tt}_{domain}.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    k1 = (tt, dim[2], dim[0])
    k2 = (tt, dim[2], dim[1])
    if k1 not in matrices:
        errors.append(f"Key k1={k1!r} not in matrices.")
        continue
    if k2 not in matrices:
        errors.append(f"Key k2={k2!r} not in matrices.")
        continue

    if engine == CReferenceEvaluator:
        if domain != "":
            continue
        if max(dim) > 256:
            # too slow
            continue
        if tt == TensorProto.FLOAT16 and max(dim) > 50:
            repeat, number = 2, 2
        if provider != ["CPUExecutionProvider"]:
            continue
        if tt not in [TensorProto.FLOAT, TensorProto.FLOAT16]:
            continue

        pbar.set_description(
            f"t={tt} e={engine.__name__} p={provider[0][:4]} dim={dim}"
        )

        feeds = {"A": matrices[k1].numpy(), "B": matrices[k2].numpy()}
        sess = engine(onx)
        sess.run(None, feeds)
        obs = measure_time(lambda: sess.run(None, feeds), repeat=repeat, number=number)

    elif engine == InferenceSession:
        if provider[0] not in get_available_providers():
            errors.append(f"provider={provider[0]} is missing")
            continue
        pbar.set_description(
            f"t={tt} e={engine.__name__} p={provider[0][:4]} dim={dim}"
        )
        opts = SessionOptions()
        r = get_ort_ext_libs()
        if r is not None:
            opts.register_custom_ops_library(r[0])
        try:
            sess = engine(onx.SerializeToString(), opts, providers=provider)
        except (NotImplemented, InvalidGraph, Fail) as e:
            # not implemented
            errors.append((tt, engine.__class__.__name__, provider, domain, e))
            continue

        if provider == ["CPUExecutionProvider"]:
            the_feeds = {"A": matrices[k1], "B": matrices[k2]}
        else:
            the_feeds = {"A": matrices_cuda[k1], "B": matrices_cuda[k2]}

        # warmup
        out_names = (
            ["C", "time"] if domain == "onnx_extented.ortops.tutorial.cuda" else ["C"]
        )
        for i in range(5):
            sess._sess.run_with_ort_values(the_feeds, out_names, None)[0]
        # benchamrk
        times = []

        def fct_benchmarked():
            got = sess._sess.run_with_ort_values(the_feeds, out_names, None)
            if len(got) > 1:
                times.append(got[1])

        obs = measure_time(fct_benchmarked, repeat=repeat, number=number)
        internal_time = None
        if len(times) > 0:
            np_times = [t.numpy() for t in times]
            internal_time = (sum(np_times) / len(times))[0]

    else:
        errors.append(f"unknown engine={engine}")
        continue

    stype = {
        TensorProto.FLOAT: "f32",
        TensorProto.FLOAT16: "f16",
        TensorProto.INT8: "i8",
        TensorProto.INT16: "i16",
        TensorProto.INT32: "i32",
        TensorProto.UINT32: "u32",
        TensorProto.FLOAT8E4M3FN: "e4m3fn",
        TensorProto.FLOAT8E5M2: "e5m2",
    }[tt]
    obs.update(
        dict(
            engine={"InferenceSession": "ort", "CReferenceEvaluator": "np"}[
                engine.__name__
            ],
            stype=stype,
            type=f"{stype}",
            M=dim[0],
            N=dim[1],
            K=dim[2],
            cost=numpy.prod(dim) * 4,
            cost_s=f"{numpy.prod(dim) * 4}-{dim[0]}x{dim[1]}x{dim[2]}",
            repeat=repeat,
            number=number,
            domain={
                "": "-",
                "com.microsoft": "ORT",
                "onnx_extented.ortops.tutorial.cuda": "EXT",
            }[domain],
            provider={
                "CPUExecutionProvider": "cpu",
                "CUDAExecutionProvider": "cuda",
            }[provider[0]],
            platform=platform.processor(),
            intime=internal_time,
        )
    )
    data.append(obs)
    if unit_test_going() and len(data) >= 2:
        break


df = DataFrame(data)
df.to_excel("plot_bench_gemm_ort.xlsx")
df.to_csv("plot_bench_gemm_ort.csv")
df.drop(["min_exec", "max_exec"], axis=1).to_csv("plot_bench_gemm_ort.csv")
print(df.head().T)
df

#####################################
# The errors
# ++++++++++
for i, e in enumerate(errors):
    print(f"{i+1}/{len(errors)}-{e}")

##############################################
# Summary
# +++++++

piv = pivot_table(
    df,
    index=["cost"],
    columns=["provider", "type", "domain", "engine"],
    values=["average", "intime"],
)
piv.reset_index(drop=False).to_excel("plot_bench_gemm_ort_summary.xlsx")
piv.reset_index(drop=False).to_csv("plot_bench_gemm_ort_summary.csv")


print("summary")
print(piv)
piv

########################################
# With the dimensions.

pivs = pivot_table(
    df,
    index=["cost_s"],
    columns=["provider", "type", "domain", "engine"],
    values=["average", "intime"],
)
print(pivs)

##############################
# plot

dfi = df[
    df.type.isin({"f32", "f16", "bf16", "e4m3fn", "e5m2"}) & df.engine.isin({"ort"})
]
pivi = pivot_table(
    dfi,
    index=["cost"],
    columns=["type", "domain", "provider", "engine"],
    values="average",
)

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
piv.plot(ax=ax[0], title="Gemm performance\nlower is better", logx=True, logy=True)
if pivi.shape[0] > 0:
    pivi.plot(
        ax=ax[1],
        title=f"Gemm performance ORT\n{platform.processor()}",
        logx=True,
        logy=True,
    )
fig.tight_layout()
fig.savefig("plot_bench_gemm_ort.png")
