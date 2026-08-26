# Quant Operator Hash Profile v1

状态：已冻结（Frozen）
版本：1.0.0
冻结日期：2026-08-26

本文件定义 Quant Operator Contract v1 摘要的唯一可互操作计算方式。
所有摘要标识均写为 `sha256:<64 lowercase hex>`。

## 1. Manifest digest

manifest `MUST` 是有效 UTF-8 文件。manifest digest 是该文件从第一个字节到
最后一个字节的原始文件 bytes 的 SHA-256，包括现有空白、字段顺序和末尾
换行。计算时 `MUST NOT` 重新序列化 JSON、执行 JSON canonicalization、转换
换行或排除字段。

manifest body `MUST NOT` 包含自己的 `manifest_digest`。该 digest `MUST` 写入
外部 binding/certification record，因此不存在自引用。

## 2. Source-tree digest

manifest 的 `implementation.source_files` 是摘要输入的完整且显式的文件集。
每个值 `MUST` 是以 provider release source-tree root 为基准的规范化相对
POSIX 路径；列表 `MUST` 非空、唯一，路径 `MUST NOT` 是绝对路径、包含反斜杠、
空 segment、`.` segment 或 `..` segment。路径解析后 `MUST` 仍位于 root 中，
并指向普通文件。

计算步骤如下：

1. 按相对 POSIX 路径 UTF-8 bytes 的字典序排序；UTF-8 保持 Unicode code point
   顺序，因此 reference implementation 直接按 Python string 排序。
2. 对每个文件的原始 bytes 计算 SHA-256，得到不带 `sha256:` 前缀的 64 位
   lowercase hex。
3. 对每项准确编码为 `<path>\0<sha256(raw file bytes)>\n`。`<path>` 使用 UTF-8，
   `\0` 是单个 NUL byte，hex 使用 ASCII，`\n` 是单个 LF byte。
4. 按排序后的顺序连接所有编码项，对连接结果计算 SHA-256。
5. 将结果写为 `sha256:<64 lowercase hex>`。

文件内容 `MUST NOT` 进行文本解码、换行转换、格式化或 canonicalization。

## 3. Implementation digest

implementation digest 是最终正式 `.whl` 文件完整原始 bytes 的 SHA-256。
它不是解压后的目录摘要，也不是 sdist、安装目录、distribution metadata
子集或 wheel 内文件摘要的组合。重新打包产生不同 wheel bytes 时，即使源码
相同，也会产生不同 implementation digest。

## 4. Contract surface and binding digests

QuantPanel schema、OperatorManifest schema、OperatorBinding schema 和
`reference_validator_v1.py` 的 digest 都与 manifest digest 使用相同的准确文件
bytes 规则。每个外部 binding/certification record 的 `contract_surface` `MUST`
分别固定这四个工件的 release 与 digest，并以 `surface_release` 标识该完整固定
元组的合并接受集合。

binding 内的 `operator_binding_schema.digest` 是 schema 文件的摘要，不是 binding
实例自身的摘要。binding `MUST NOT` 包含自身文件摘要；binding instance digest
如有需要，只能记录在更外层的 registry 或 certification envelope 中。

每个 binding 还 `MUST` 固定 manifest digest、source-tree digest、implementation
digest、distribution version 和完整 source commit。

## 5. Reference functions

`reference_validator_v1.py` 发布以下参考函数：

- `sha256_file(path)`：计算准确文件 bytes 的摘要，可用于 manifest、三个 schema、
  `reference_validator_v1.py` 和正式 wheel。
- `sha256_source_tree(root, source_files)`：执行第 2 节 source-tree profile。

其他语言可以重新实现这些函数，但 conformance vectors 的结果 `MUST` 与参考
实现逐 byte 一致。
