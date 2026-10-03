import re
from enum import Enum
from collections import defaultdict, OrderedDict

# 错误类型优先级
class VerusErrorType(Enum):
    # 语法结构错误
    UnexpectedToken = 1      
    ExpectedCurlyBraces = 2  
    ExpectedComma = 3         
    BadQuantifierSyntax = 4   

    UnresolvedImport = 5     
    CantFindValue = 6      

    MismatchedType = 7     
    IncompatibleTypes = 8
    TypeAnnotationNeeded = 9  

    TriggerInferenceFail = 10 
    UnknownProver = 11     

    Other = 12

    # 逻辑验证错误
    ArithmeticFlow = 13
    PreCondFail = 14
    InvFailEnd = 15
    InvFailFront = 16
    AssertFail = 17
    PostCondFail = 18   

# 解析verus报错信息
def parse_verus_errors(output: str):
    errors = []

    # 错误类型关键字映射
    verus_error_types = {
        "mismatched types": "MismatchedType",
        "type annotations needed": "TypeAnnotationNeeded",
        "incompatible types": "IncompatibleTypes",
        "cannot find value": "CantFindValue",
        "unresolved import": "UnresolvedImport",
        "invariant not satisfied before loop": "InvFailFront",
        "invariant not satisfied at end of loop body": "InvFailEnd",
        "precondition not satisfied": "PreCondFail",
        "postcondition not satisfied": "PostCondFail",
        "assertion failed": "AssertFail",
        "possible arithmetic underflow/overflow": "ArithmeticFlow",
        "expected curly braces": "ExpectedCurlyBraces",
        "expected `,`": "ExpectedComma",
        "unexpected token": "UnexpectedToken",
        "forall, choose, and exists do not allow parentheses": "BadQuantifierSyntax",
        "Could not automatically infer triggers": "TriggerInferenceFail",
        "unknown prover": "UnknownProver",
    }

    # 捕获 error + location
    header_pattern = re.compile(
        r'(error(?:\[[^\]]+\])?:[^\n]+)\n\s*-->\s*(.+?):(\d+):(\d+)',
        re.MULTILINE
    )

    # 捕获代码箭头行（带 | 和 --^）
    code_block_pattern = re.compile(
        r'^\s*\|\n(?:(?:\s*\d+\s*\|.*\n)|(?:\s*\|\s*.*\n))+',
        re.MULTILINE
    )

    pos = 0
    while True:
        header_match = header_pattern.search(output, pos)
        if not header_match:
            break

        error_type_msg = header_match.group(1).strip()
        file_path = header_match.group(2).strip()
        line = int(header_match.group(3))
        column = int(header_match.group(4))
        pos = header_match.end()

        # 尝试匹配紧跟在 header 后的代码块
        code_match = code_block_pattern.search(output, pos)
        code_block = ""
        message_detail = ""
        if code_match and code_match.start() <= pos + 5:  # 放宽匹配
            code_block = code_match.group().rstrip()
            pos = code_match.end()

            # 去掉前导 | 并作为 message_detail
            lines = code_block.split("\n")
            message_lines = [l[1:] if l.startswith("|") else l for l in lines if l.strip()]
            message_detail = "\n".join(message_lines)

        # 拆分 header 中原始 error type 和第一行 msg
        if ": " in error_type_msg:
            original_type, first_msg = error_type_msg.split(": ", 1)
        else:
            original_type = error_type_msg
            first_msg = ""

        # message = header msg + 详细信息
        message = first_msg
        if message_detail:
            message += "\n" + message_detail

        # 根据 message 内容匹配 error_type
        matched_type = "Other"  # 默认
        for keyword, vtype in verus_error_types.items():
            if keyword.lower() in message.lower():
                matched_type = vtype
                break

        errors.append({
            "error_type": matched_type,
            "message": message.strip(),
            "line": line,
            "column": column,
            "code": code_block,
        })

    return errors

# 对parse_verus_errors的输出按照error_type分组
def group_errors_by_type(errors):
    grouped = defaultdict(list)
    for err in errors:
        grouped[err['error_type']].append(err)
    return dict(grouped)

# 对group_errors_by_type的输出按照优先级排序
def sort_grouped_errors_by_priority(grouped_errors):
    def get_priority(et):
        return VerusErrorType[et].value if et in VerusErrorType.__members__ else VerusErrorType.Other.value

    # 按优先级排序 key
    sorted_keys = sorted(grouped_errors.keys(), key=get_priority)

    # 构造新的有序字典
    sorted_grouped = OrderedDict()
    for k in sorted_keys:
        sorted_grouped[k] = grouped_errors[k]

    return sorted_grouped
