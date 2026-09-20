# 02.1 — Add add_inventory / remove_inventory Tools

> Commit: `feat: add add_inventory / remove_inventory tools`
> Tag: `v0.2.1-multi-tool-registry`

## 1. Problem

到 Phase 1 为止只有一个只读工具 `get_inventory`。真实业务里"查库存"只是最基础的操作，还要能"入库"、"出库"——从只读扩展到读写，也第一次让 `agent.py` 里的 `TOOLS`/`TOOL_FUNCTIONS` 真正成为一份"注册表"而不是只有一条记录的摆设。这一步顺带为以后的 Permission 章节埋下伏笔：READ 操作（`get_inventory`）和 WRITE 操作（`add_inventory`/`remove_inventory`）现在开始有明确的区分，虽然这一章还不做权限控制。

## 2. Minimal Example

`src/tools/inventory.py` 新增两个函数，故意保持和 `get_inventory` 一样简单：

```python
def add_inventory(product_id: str, qty: int) -> dict | None:
    product = products.get(product_id)
    if product is None:
        return None
    product["stock"] += qty
    return product


def remove_inventory(product_id: str, qty: int) -> dict | None:
    product = products.get(product_id)
    if product is None:
        return None
    product["stock"] -= qty
    return product
```

**这一版故意不做任何校验**——`qty` 可以传负数，`remove_inventory` 可以把库存减成负数，都不会报错。这不是遗漏，是刻意的范围控制：参数校验是 Phase 3（Tool Validation）的工作，这一步只解决"写操作本身怎么接进 Tool Calling 流程"这一件事。为了不让这个"故意的洞"以后被忘记，专门写了一条测试把当前行为钉死（见第 6 节）。

对应的 `ADD_INVENTORY_SCHEMA`/`REMOVE_INVENTORY_SCHEMA` 结构和 `GET_INVENTORY_SCHEMA` 一致，多了一个必填的 `qty: integer` 参数。

`src/agent/agent.py` 的 Registry 从一条记录变成三条：

```python
TOOLS = [GET_INVENTORY_SCHEMA, ADD_INVENTORY_SCHEMA, REMOVE_INVENTORY_SCHEMA]
TOOL_FUNCTIONS = {
    "get_inventory": get_inventory,
    "add_inventory": add_inventory,
    "remove_inventory": remove_inventory,
}
```

`ask()` 本身**一行代码都没有改动**——这是 Commit 1.2 设计 `TOOLS`/`TOOL_FUNCTIONS` 这一对列表/字典时就想好的：加一个新工具，只需要在这两个地方各加一条，函数本身不需要认识任何具体的工具名字，靠 `tool_call.function.name` 去 `TOOL_FUNCTIONS` 里查。

## 3. Break It / Observe / Why It Failed

不适用——这一步没有引入新的失败模式，`add_inventory`/`remove_inventory` 目前和 `get_inventory` 一样，"正常路径能走通"就是这一步的目标，故意留下的校验缺口会在 Phase 3 被系统性地测试到。

## 4. Implement the Fix

不适用，理由同上。

## 5. Introduce Industrial Solution

不适用，第一次框架对比在 Phase 17。

## 6. Evaluate

```bash
pytest -v
```

实际输出：

```text
tests/test_agent.py::test_tool_functions_keys_match_tool_schema_names PASSED
tests/test_inventory.py::TestAddInventory::test_increases_stock_by_qty PASSED
tests/test_inventory.py::TestAddInventory::test_returns_none_for_unknown_product PASSED
tests/test_inventory.py::TestAddInventory::test_schema_requires_product_id_and_qty PASSED
tests/test_inventory.py::TestRemoveInventory::test_decreases_stock_by_qty PASSED
tests/test_inventory.py::TestRemoveInventory::test_returns_none_for_unknown_product PASSED
tests/test_inventory.py::TestRemoveInventory::test_schema_requires_product_id_and_qty PASSED
tests/test_inventory.py::TestRemoveInventory::test_does_not_validate_quantity_yet PASSED
...（加上此前的 24 个）

32 passed in 0.55s
```

`test_does_not_validate_quantity_yet` 这条值得单独说一下：它断言"出库 999 个会让库存变成负数"——这不是在测试一个我们希望存在的行为，是在把当前**故意留下的缺口**钉死成一条可以被回归检测到的事实。等 Phase 3 真的加上校验之后，这条测试应该会失败，那时候需要同步更新它（改成断言"校验会拒绝这个请求"），而不是删掉——失败本身就是"校验生效了"的证明。

另外新增了 `_restore_products` 这个 `autouse` fixture——`add_inventory`/`remove_inventory` 会真的修改模块级的 `products` 字典，如果不在每个测试后还原，测试跑的先后顺序会互相污染彼此的库存数字（比如后跑的测试断言 `stock == 20`，结果因为前面的测试已经改过它而失败）。这是这本教程第一次遇到"测试之间共享可变状态"这个问题。

顺带加了一条 `test_tool_functions_keys_match_tool_schema_names`——断言 `TOOLS` 里每个 schema 的 `function.name` 和 `TOOL_FUNCTIONS` 的 key 完全对得上。工具数量还小的时候手动核对没问题，工具一多，改名字漏改一边的问题会越来越容易发生，这条测试从这一步开始覆盖任意数量的工具，不用每加一个工具就手动改一遍断言。

DoD 要求"三个工具都能被模型正确选择和调用"，靠真实调用验证：

```python
print(ask("A100 还有多少库存？").text)          # get
print(ask("刚到货 5 个 A100，帮我入库。").text)   # add
print(ask("卖出去 3 个 A100，帮我出库。").text)   # remove
```

实际输出（`deepseek-chat`，2026-09-20，库存从 20 → 25 → 22，数字和入库/出库的数量精确对上）：

```text
--- get ---
A100（工业传感器 A100）目前还有 **20 件**库存。

--- add ---
入库完成 ✅
- 产品：Industrial Sensor A100
- 本次入库：5 个
- 当前库存：25 个

--- remove ---
出库完成 ✅
- 商品：Industrial Sensor A100
- 出库数量：3 个
- 剩余库存：22 个
```

模型三次都正确选中了对应的工具，`qty` 参数也和自然语言里的数字精确对上，没有出现选错工具或参数解析错误的情况。

## 7. Exercises

1. 故意把 `TOOL_FUNCTIONS` 里 `add_inventory` 的 key 改成和 Schema 里的 `name` 不一致（比如改成 `"addInventory"`），验证 `test_tool_functions_keys_match_tool_schema_names`（这一步新加的）能不能抓住这个问题——它现在覆盖了全部三个工具，不用手动改测试代码。
2. 问一个模糊的指令，比如"处理一下 A100"，不说清楚是入库还是出库，观察模型会怎么反应（拒绝执行、要求澄清，还是猜一个方向）。
3. 连续问两个不同商品的库存（比如"A100 和 B200 分别还有多少库存？"），观察当前的 `ask()`——它只处理"最多一次工具调用"——会发生什么。这是后面 Agent Loop 要解决的问题的一个具体预演。

## 8. What We Learned

`TOOLS`/`TOOL_FUNCTIONS` 这一对数据结构在 Commit 1.2 就设计对了，加两个新工具完全不需要改 `ask()` 本身；"故意留下的缺口"最好用一条测试钉死而不是只写在注释里，这样以后真正修复时会有一个明确的失败信号提醒你去更新它；测试之间共享可变模块状态（`products` 字典）第一次成为需要处理的问题，用 `autouse` fixture 解决；以及工具数量变多之后，`TOOLS`/`TOOL_FUNCTIONS` 两边名字是否一致，值得单独用一条测试守住，而不是靠人工核对。
