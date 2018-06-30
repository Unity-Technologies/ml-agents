# インストール

ML-Agentsをご使用いただくためには、まず最初に下記の物をインストールしていただく必要がございます。

1. Unity
2. Python

各項目に関しての実際の手順に関しては下記に記載をいたしましたので、適宜ご覧ください。

## **Unity 2017.1** 以降のバージョンのインストールについて

こちらより、Unityの方を[Download](https://store.unity.com/download)とinstallしてください。
またもし、Dokcerをご使用される場合は、下記の項目の選択をお忘れないようにしてください。

_Linux Build Support_

<p align="center">
    <img src="images/unity_linux_build_support.png" 
        alt="Linux Build Support" 
        width="500" border="10" />
</p>

## Ml-Agents レポのクローン

Unityがインストールできましたら下記より、ML-Agentsのレポをクローンして下さい。 

    git clone https://github.com/Unity-Technologies/ml-agents.git
    
ディレクトリ構成の簡単な説明についてですが、
- `unity-environment` のディレクトリの中にはAsstsとProjectSettingsが入っております。
こちらのAssetsの方を今後作成される皆様のProjectの方でimportしていただいてご活用ください。
- `python`ディレクトリに関してはトレーニングのコードやモデルを格納するためのディレクトリ、
GRPCを使用したUnityEnvironmentとのコミューケーションサポートアプリケーション等が含まれております。 

## Pythonのインストールについて

ML-Agentsを使用する際に必要となるのはPython3.5か3.6でございます。 
また、ご存知かもしれませんが、Pythonでは各Projectsごとにソースコードで使用されているライブラリの方を特殊なファイルに書き出すことができます。
ですので、Pythonインストール後に、一度先ほどcloneしていただいたML-Agentsのディレクトリに移動していただき、下記の`requirements.txt`を
`pip3 install -r requirements.txt`でご自分の環境に入れてください。下記に実際のファイルへの参照先ですので、
インストールしているライブラリを確認いただくのもいいかもしれません。  
[requirements file](../python/requirements.txt)  

また、もし、皆様のPythonの環境でPipがインストールされていなければ下記を参考にしてください。
[instructions](https://packaging.python.org/guides/installing-using-linux-tools/#installing-pip-setuptools-wheel-with-linux-package-managers)

こちらが必須のアイテムとなります。
- [TensorFlow](Background-TensorFlow.md)
- [Jupyter](Background-Jupyter.md)

### インストール方法一覧
- Windows : [Installation guide](Installation-Windows.md)
- Mac/Unix : [Installation guide](https://www.python.org/downloads/)
- Docker-based Installation : [Installation guide](Using-Docker.md). 

## Next Steps
次のステップとしましては、下記のBasic Guideから3D BallゲームでのPPOアルゴリズムを適用してのエージェントの育成をしてみることができます。
こちらのTutorialを通して、実際の環境の構築や、トレーニングに必要となる重要なML-Agentsの概念を説明していきたいと思います。

[Basic Guide](Basic-Guide.md)

## Help
もしご不明点がございましたら下記を参照ください。

- [FAQ](FAQ.md)
- [Limitations](Limitations.md)

また、上記に記載のないトラブル等に関してはGithub上からIssueを挙げていただくことができます。
- [submit an issue](https://github.com/Unity-Technologies/ml-agents/issues)
ご使用の環境やエラーの内容の記述など、なるべく多くの情報を記載いただくと問題解決に至るまでが速くなりますのでご承知おきください・

