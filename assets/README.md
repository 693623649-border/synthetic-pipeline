# Zoo-Bus reference asset pack

将参考项目的 `scene_gen/src` 目录复制到这里，或运行时通过：

```bash
python3 -m synthetic_vqa smoke \
  --asset-root /path/to/scene_gen/src \
  --strict-assets
```

渲染器要求以下文件：

```text
background.jpeg
bench.png
person.png
stopSign.png
zebra.png
elephant.png
giraffe.png
clock.png
```

当前仓库不内置参考数据集的私有像素资产。未提供 `--asset-root` 时，smoke
模式使用内存兼容资产，用于验证 alpha composite、原生/输出 bbox、编号叠加、
JPEG 导出和状态保存；这不会声称与参考数据集逐像素相同。

