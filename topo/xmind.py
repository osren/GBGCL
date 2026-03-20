import xmindparser
import os
import json


def xmind_to_markdown(xmind_file, output_file=None):
    try:
        data = xmindparser.xmind_to_dict(xmind_file)
        # 调试代码：打印解析后的数据结构
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"解析 XMind 文件失败: {e}")
        return

    if not data:
        print("XMind 文件为空或解析失败")
        return

    output_file = output_file or f"{os.path.splitext(xmind_file)[0]}.md"
    markdown_content = generate_markdown(data)

    if not markdown_content.strip():
        print("警告：生成的 Markdown 内容为空，请检查 XMind 文件结构")

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"转换成功，文件已保存到: {output_file}")
    except Exception as e:
        print(f"写入失败: {e}")


def generate_markdown(data):
    markdown = ""
    if isinstance(data, list):
        for sheet in data:
            markdown += f"# {sheet.get('title', '未命名 Sheet')}\n\n"
            if 'topic' in sheet:
                markdown += process_topic(sheet['topic'], 2)
    return markdown


def process_topic(topic, level):
    content = ""
    title = topic.get('title', '无标题')
    content += f"{'#' * level} {title}\n\n"

    if 'note' in topic:
        content += f"> {topic['note']}\n\n"

    if 'labels' in topic and topic['labels']:
        content += "​**标签**: " + ", ".join(topic['labels']) + "\n\n"

    if 'topics' in topic:
        for subtopic in topic['topics']:
            content += process_topic(subtopic, level + 1)

    return content


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python xmind_to_md.py <xmind文件> [输出md文件]")
        sys.exit(1)
    xmind_to_markdown(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)