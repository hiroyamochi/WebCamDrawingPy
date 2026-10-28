// WebCam Drawing の起動用 exe（Windows 標準の .NET Framework の csc でビルドする）
// 埋め込んだ payload.zip（Python 公式の embeddable Python + アプリ）を初回だけ
// %LOCALAPPDATA%\WebCamDrawing\<ビルドID> に展開し、署名付きの pythonw.exe でアプリを起動する。
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Threading;
using System.Windows.Forms;

static class Launcher
{
    const string Title = "WebCam Drawing";

    [STAThread]
    static int Main()
    {
        try
        {
            var asm = Assembly.GetExecutingAssembly();
            string root = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "WebCamDrawing");
            // ビルドごとに変わる ID で展開先を分け、更新後は自動で展開し直す
            string dir = Path.Combine(root, asm.ManifestModule.ModuleVersionId.ToString("N"));
            if (!Directory.Exists(dir))
                ExtractWithProgress(asm, root, dir);

            Process.Start(new ProcessStartInfo
            {
                FileName = Path.Combine(dir, @"python\pythonw.exe"),
                Arguments = "\"" + Path.Combine(dir, @"app\webcamdrawing.py") + "\"",
                WorkingDirectory = dir,
                UseShellExecute = false,
            });
            return 0;
        }
        catch (Exception e)
        {
            MessageBox.Show(e.ToString(), Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }

    // 展開には数十秒かかるので、その間は小さな待機画面を出す
    static void ExtractWithProgress(Assembly asm, string root, string dir)
    {
        Application.EnableVisualStyles();
        var form = new Form
        {
            Text = Title,
            Width = 360,
            Height = 90,
            FormBorderStyle = FormBorderStyle.FixedDialog,
            ControlBox = false,
            StartPosition = FormStartPosition.CenterScreen,
        };
        var bar = new ProgressBar { Style = ProgressBarStyle.Marquee, Dock = DockStyle.Fill };
        form.Controls.Add(bar);

        Exception error = null;
        form.Shown += delegate
        {
            new Thread(() =>
            {
                try { Extract(asm, root, dir); }
                catch (Exception e) { error = e; }
                form.BeginInvoke((Action)form.Close);
            }) { IsBackground = true }.Start();
        };
        Application.Run(form);
        if (error != null)
            throw error;
    }

    static void Extract(Assembly asm, string root, string dir)
    {
        // 途中で中断されても壊れたフォルダを使わないよう、一時フォルダに展開してから移す
        string tmp = dir + ".tmp";
        if (Directory.Exists(tmp))
            Directory.Delete(tmp, true);
        using (var stream = asm.GetManifestResourceStream("payload.zip"))
        using (var zip = new ZipArchive(stream))
            zip.ExtractToDirectory(tmp);
        Directory.Move(tmp, dir);

        // 古いバージョンの展開先を片付ける（使用中なら次回に回す）
        foreach (string old in Directory.GetDirectories(root))
        {
            if (string.Equals(old, dir, StringComparison.OrdinalIgnoreCase))
                continue;
            try { Directory.Delete(old, true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }
}
