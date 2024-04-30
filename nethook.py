"""
これらのユーティリティは、デバッグ、可視化、モデルの動作の細かな制御など、さまざまな目的でPyTorchモデルを調査、変更、操作するための便利な方法を提供します。
特に、`Trace`と`TraceDict`クラスは、モデルの特定のレイヤーの中間出力と勾配を調べるための強力なツールであり、モデルの動作を理解しデバッグするのに非常に役立ちます。
`subsequence`関数は、Sequentialモデルからサブモデルを作成することを可能にし、転移学習やモデルの改変に便利です。
`get_module`、`replace_module`、`get_parameter`関数は、ドット表記を使用してモデルの特定の部分にアクセスおよび変更するための便利な方法を提供します。
最後に、`set_requires_grad`関数は、訓練やファインチューニング中にモデルの一部を凍結または解凍するのに便利です。
"""
import contextlib
import copy
import inspect
from collections import OrderedDict
import torch

"""
1. `Trace`クラス:
  - このクラスは、PyTorchモデルの1つのレイヤーを一度にフックするために使用されます。
  - モデルの順伝播中に、指定されたレイヤーの出力（オプションで入力も）を保持します。
  - `with`文によるコンテキストマネージャーとして使用できます。
  - レイヤーの出力をクローン、デタッチ、または勾配を保持するオプションを提供します。
  - レイヤーの出力を変更してからモデルの残りの部分に渡すこともできます。
  - `StopForward`例外を発生させることで、特定のレイヤーで順伝播を停止するために使用できます。
"""
class Trace(contextlib.AbstractContextManager):
    """
    named layerの出力を保持するには、ネットワークの計算中に以下のようにします。
        with Trace(net, 'layer.name') as ret:
            _ = net(inp)
            representation = ret.output
    レイヤー名なしでレイヤーモジュールを直接渡すこともでき、その出力が保持されます。
    デフォルトでは出力オブジェクトへの直接の参照が返されますが、オプションでこれを制御できます。
        clone=True  - 出力のコピーを保持します。これは、ネットワークによって後で出力が
                      インプレースで変更される可能性がある場合に便利です。
        detach=True - detachされた参照またはコピーを保持します。
                      （デフォルトでは、値はグラフにアタッチされたままになります）
        retain_grad=True - 出力の勾配を保持するよう要求します。
                            backward()の後、ret.output.gradが更新されます。
        retain_input=True - 入力も保持します。
        retain_output=False - 出力の保持を無効にできます。
        edit_output=fn - レイヤーの出力をモデルの残りの部分に渡す前に変更するための関数を呼び出します。
                          fnは、元の出力とレイヤー名の(output, layer)引数を受け取ることができます。
        stop=True - レイヤーが実行された後、StopForward例外をスローします。
                    これにより、モデルの一部のみを実行できます。
    """
    def __init__(
        self,
        module,
        layer=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        """
        forward メソッドを、呼び出しをインターセプトし、フックを追跡するクロージャで置き換えるメソッド。
        """
        retainer = self
        self.layer = layer
        if layer is not None:
            # print("romeworkspace/rome/util/nethook.py, line 69")
            module = get_module(module, layer)
        def retain_hook(m, inputs, output):
            if retain_input:
                retainer.input = recursive_copy(
                    inputs[0] if len(inputs) == 1 else inputs,
                    clone=clone,
                    detach=detach,
                    retain_grad=False,
                )  # retain_gradは出力にのみ適用されます。
            if edit_output:
                output = invoke_with_optional_args(
                    edit_output, output=output, layer=self.layer
                )
            if retain_output:
                retainer.output = recursive_copy(
                    output, clone=clone, detach=detach, retain_grad=retain_grad
                )
                # retain_gradが設定されている場合、自明なコピー操作も挿入します。
                # これにより、インプレースな操作をエラーなしで続けることができます。
                if retain_grad:
                    output = recursive_copy(retainer.output, clone=True, detach=False)
            if stop:
                raise StopForward()
            return output
        self.registered_hook = module.register_forward_hook(retain_hook)
        self.stop = stop
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
    def close(self):
        self.registered_hook.remove()

"""
2. `TraceDict`クラス:
  - このクラスは、PyTorchモデルの複数のレイヤーを一度にフックするために使用されます。
  - `OrderedDict`のサブクラスであり、コンテキストマネージャーとして使用できます。
  - モデルとフックするレイヤー名のリストを引数に取ります。
  - 指定された各レイヤーに対して`Trace`オブジェクトを作成し、辞書に格納します。
  - 出力のクローン、デタッチ、勾配の保持、出力の変更について、`Trace`と同じオプションを提供します。
"""
class TraceDict(OrderedDict, contextlib.AbstractContextManager):
    """
    指定されたネットワークの計算中に、複数の名前付きレイヤーの出力を保持するには：
        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output
    edit_outputが提供されている場合は、出力とレイヤー名の2つの引数を取る関数である必要があります。
    そして、変更された出力を返します。
    他の引数はTraceと同じです。stopがTrueの場合、リストされた最後のレイヤーの後で
    ネットワークの実行が停止します（それが最後に実行されるわけではない場合でも）。
    """
    def __init__(
        self,
        module,
        layers=None,
        retain_output=True,
        retain_input=False,
        clone=False,
        detach=False,
        retain_grad=False,
        edit_output=None,
        stop=False,
    ):
        self.stop = stop
        def flag_last_unseen(it):
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev
        for is_last, layer in flag_last_unseen(layers):
            self[layer] = Trace(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )
    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
    def close(self):
        for layer, trace in reversed(self.items()):
            trace.close()

class StopForward(Exception):
    """
    ネットワークを実行して必要な出力が保持されたサブモジュールのみの場合、
    Trace(submodule, stop=True)は、保持されたサブモジュールの直後に
    StopForward()例外を発生させて実行を即座に停止します。
    Traceがコンテキストマネージャーとして使用される場合、その例外をキャッチし、
    次のように使用できます。
    with Trace(net, layername, stop=True) as tr:
        net(inp) # layernameまでネットワークを実行するだけ
    print(tr.output)
    """
    pass

def recursive_copy(x, clone=None, detach=None, retain_grad=None):
    """
    テンソルへの参照、またはテンソルを含むオブジェクトをコピーします。
    オプションでテンソルをデタッチおよびクローンします。
    retain_gradがtrueの場合、元のテンソルにはグラデーションが保持されるようにマークされます。
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."

"""
3. `subsequence`関数:
  - この関数は、PyTorch Sequentialモデルのサブシーケンスを作成するために使用されます。
  - 指定された開始レイヤーと終了レイヤー（両端を含む）の間のモジュールのみを含むようにモデルをスライスします。
  - サブシーケンスのモジュールとパラメータをコピーするか、元のモデルと重みを共有するかを選択できます。
"""
def subsequence(
    sequential,
    first_layer=None,
    last_layer=None,
    after_layer=None,
    upto_layer=None,
    single_layer=None,
    share_weights=False,
):
    """
    PyTorch Sequentialモデルのサブシーケンスを作成し、サブシーケンスのモジュールとパラメータを
    コピーします。first_layerからlast_layer（inclusive）までのモジュールのみが含まれます。
    または、after_layerとupto_layer（exclusive）の間のモジュールが含まれます。
    すべての参照がネストされたSequentialモデル内にある限り、ドット付きレイヤー名への
    降下を処理します。

    share_weightsがTrueの場合、元のモジュールとそのパラメータを参照しますが、コピーはしません。
    それ以外の場合、デフォルトでは、完全に新しいコピーを作成します。
    """
    assert (single_layer is None) or (
        first_layer is last_layer is after_layer is upto_layer is None
    )
    if single_layer is not None:
        first_layer = single_layer
        last_layer = single_layer
    first, last, after, upto = [
        None if d is None else d.split(".")
        for d in [first_layer, last_layer, after_layer, upto_layer]
    ]
    return hierarchical_subsequence(
        sequential,
        first=first,
        last=last,
        after=after,
        upto=upto,
        share_weights=share_weights,
    )

def hierarchical_subsequence(
    sequential, first, last, after, upto, share_weights=False, depth=0
):
    """
    subsequence()の再帰的なヘルパー関数で、ドット付きレイヤー名への降下をサポートします。
    このヘルパーでは、first、last、after、uptoはドットで分割された結果の名前の配列です。
    ネストされたSequentialにのみ降下できます。
    """
    assert (last is None) or (upto is None)
    assert (first is None) or (after is None)
    if first is last is after is upto is None:
        return sequential if share_weights else copy.deepcopy(sequential)
    assert isinstance(sequential, torch.nn.Sequential), (
        ".".join((first or last or after or upto)[:depth] or "arg") + " not Sequential"
    )
    including_children = (first is None) and (after is None)
    included_children = OrderedDict()
    # A = 現在のレベルのAの短縮名。
    # AN = 最も内側でない場合の再帰的降下のためのフルネーム。
    (F, FN), (L, LN), (A, AN), (U, UN) = [
        (d[depth], (None if len(d) == depth + 1 else d))
        if d is not None
        else (None, None)
        for d in [first, last, after, upto]
    ]
    for name, layer in sequential._modules.items():
        if name == F:
            first = None
            including_children = True
        if name == A and AN is not None:  # リーフでない場合はFと同様。
            after = None
            including_children = True
        if name == U and UN is None:
            upto = None
            including_children = False
        if including_children:
            # AR = 名前が一致する場合の再帰的降下のためのフルネーム。
            FR, LR, AR, UR = [
                n if n is None or n[depth] == name else None for n in [FN, LN, AN, UN]
            ]
            chosen = hierarchical_subsequence(
                layer,
                first=FR,
                last=LR,
                after=AR,
                upto=UR,
                share_weights=share_weights,
                depth=depth + 1,
            )
            if chosen is not None:
                included_children[name] = chosen
        if name == L:
            last = None
            including_children = False
        if name == U and UN is not None:  # リーフでない場合はLと同様。
            upto = None
            including_children = False
        if name == A and AN is None:
            after = None
            including_children = True
    for name in [first, last, after, upto]:
        if name is not None:
            raise ValueError("Layer %s not found" % ".".join(name))
    # 最も外側のレベル以外では空のサブシーケンスを省略します。
    # 最も外側のレベルではNoneを返すべきではありません。
    if not len(included_children) and depth > 0:
        return None
    result = torch.nn.Sequential(included_children)
    result.training = sequential.training
    return result

"""
5. `set_requires_grad`関数:
  - この関数は、1つ以上のモデルのすべてのパラメータの`requires_grad`属性を再帰的に設定するために使用されます。
  - ブール値と1つ以上のモデルを引数として取ります。
  - 指定された値に基づいて、提供されたモデルのすべてのパラメータの`requires_grad`を設定します。
"""
def set_requires_grad(requires_grad, *models):
    """
    渡されたモデル内のすべてのパラメータに対して、requires_gradをtrueまたはfalseに設定します。
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)

"""
4. `get_module`、`replace_module`、`get_parameter`関数:
  - これらの関数は、ドット表記で指定されたモジュールおよびパラメータ名を解決するために使用されます。
  - `get_module`は、ドット表記の名前に基づいてモデルからモジュールを取得します。
  - `replace_module`は、ドット表記の名前に基づいて、モデル内のモジュールを新しいモジュールに置き換えます。
  - `get_parameter`は、ドット表記の名前に基づいてモデルからパラメータを取得します。
"""
def get_module(model, name):
    """
    指定されたモデル内の名前付きモジュールを見つけます。
    """
    # print(name)
    for n, m in model.named_modules():
        # print(n)
        if n == name:
            return m
    raise LookupError(name)
def get_parameter(model, name):
    """
    指定されたモデル内の名前付きパラメータを見つけます。
    """
    # print(model.named_parameters())
    for n, p in model.named_parameters():
        # print(n)
        # print(name)
        if n == name:
            return p
    raise LookupError(name)
def replace_module(model, name, new_module):
    """
    指定されたモデル内の名前付きモジュールを置き換えます。
    """
    if "." in name:
        parent_name, attr_name = name.rsplit(".", 1)
        model = get_module(model, parent_name)
    # original_module = getattr(model, attr_name)
    setattr(model, attr_name, new_module)


def invoke_with_optional_args(fn, *args, **kwargs):
    """
    関数を、その関数が受け入れるように書かれた引数のみで呼び出します。
    名前で一致する引数を優先し、次の規則を使用します。
    (1) 一致する名前の引数は名前で渡されます。
    (2) 残りの名前が一致しない引数は、順番で渡されます。
    (3) 関数が受け入れられない余分な呼び出し元の引数は渡されません。
    (4) 呼び出し元が提供できない余分な必須の関数引数は、TypeErrorを発生させます。
    通常のPythonの呼び出し規約は、呼び出し元に新しい引数を渡すことを要求せずに、
    新しいバージョンで追加の引数を受け入れるように改訂される可能性のある関数を
    サポートするのに役立ちます。この関数は、被呼び出し側がそれらの新しい引数を
    受け入れることを要求せずに、追加の引数を提供するように改訂される可能性のある
    関数の呼び出し元をサポートするのに役立ちます。
    """
    argspec = inspect.getfullargspec(fn)
    pass_args = []
    used_kw = set()
    unmatched_pos = []
    used_pos = 0
    defaulted_pos = len(argspec.args) - (
        0 if not argspec.defaults else len(argspec.defaults)
    )
    # 最初に名前で一致する位置引数を渡し、次に位置で渡します。
    for i, n in enumerate(argspec.args):
        if n in kwargs:
            pass_args.append(kwargs[n])
            used_kw.add(n)
        elif used_pos < len(args):
            pass_args.append(args[used_pos])
            used_pos += 1
        else:
            unmatched_pos.append(len(pass_args))
            pass_args.append(
                None if i < defaulted_pos else argspec.defaults[i - defaulted_pos]
            )
    # 一致しない位置引数を、一致しないキーワード引数で順番に埋めます。
    if len(unmatched_pos):
        for k, v in kwargs.items():
            if k in used_kw or k in argspec.kwonlyargs:
                continue
            pass_args[unmatched_pos[0]] = v
            used_kw.add(k)
            unmatched_pos = unmatched_pos[1:]
            if len(unmatched_pos) == 0:
                break
        else:
            if unmatched_pos[0] < defaulted_pos:
                unpassed = ", ".join(
                    argspec.args[u] for u in unmatched_pos if u < defaulted_pos
                )
                raise TypeError(f"{fn.__name__}() cannot be passed {unpassed}.")
    # 受け入れ可能な場合は、残りのキーワード引数を渡します。
    pass_kw = {
        k: v
        for k, v in kwargs.items()
        if k not in used_kw and (k in argspec.kwonlyargs or argspec.varargs is not None)
    }
    # 受け入れ可能な場合は、残りの位置引数を渡します。
    if argspec.varargs is not None:
        pass_args += list(args[used_pos:])
    return fn(*pass_args, **pass_kw)