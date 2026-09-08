package com.research.pilotcontroller;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Base64;
import android.webkit.JavascriptInterface;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {

    private static final String KAGGLE_USER = "vaibhavchavanpatil";
    private static final String KAGGLE_KEY = "9c5eb49cd4218f2adfe038c570bfdf2e";

    private WebView webView;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private final Handler pollHandler = new Handler(Looper.getMainLooper());

    private static final String HTML_PAGE = "<!DOCTYPE html>" +
            "<html><head><meta name='viewport' content='width=device-width, initial-scale=1.0, user-scalable=no'>" +
            "<style>" +
            "  * { box-sizing: border-box; margin: 0; padding: 0; }" +
            "  body { background: #070d18; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 16px; padding-bottom: 30px; }" +
            "  .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }" +
            "  .title-wrap { display: flex; flex-direction: column; }" +
            "  .title { font-size: 20px; font-weight: 800; color: #ffffff; display: flex; align-items: center; gap: 8px; }" +
            "  .subtitle { font-size: 11px; color: #94a3b8; margin-top: 3px; }" +
            "  .live-pill { display: flex; align-items: center; gap: 6px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; padding: 4px 10px; border-radius: 20px; font-size: 10.5px; font-weight: 700; color: #34d399; }" +
            "  .pulse-dot { width: 7px; height: 7px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; animation: pulse 1.5s infinite; }" +
            "  @keyframes pulse { 0% { opacity: 0.3; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.3); } 100% { opacity: 0.3; transform: scale(0.9); } }" +
            "  .card { background: rgba(26, 38, 57, 0.7); border: 1px solid #2d3f59; border-radius: 14px; padding: 14px; margin-bottom: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.4); backdrop-filter: blur(10px); }" +
            "  .card-title { font-size: 13px; font-weight: 700; color: #cbd5e1; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }" +
            "  .progress-bg { background: #0f172a; height: 12px; border-radius: 6px; overflow: hidden; margin: 8px 0; border: 1px solid #334155; }" +
            "  .progress-fill { background: linear-gradient(90deg, #10b981, #3b82f6); height: 100%; width: 0%; transition: width 0.5s ease-in-out; border-radius: 6px; }" +
            "  .badge { font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 12px; text-transform: uppercase; letter-spacing: 0.5px; display: inline-flex; align-items: center; gap: 4px; }" +
            "  .badge-running { background: rgba(59, 130, 246, 0.25); color: #60a5fa; border: 1px solid #3b82f6; animation: glow 2s infinite; }" +
            "  @keyframes glow { 0% { box-shadow: 0 0 4px rgba(59,130,246,0.3); } 50% { box-shadow: 0 0 10px rgba(59,130,246,0.7); } 100% { box-shadow: 0 0 4px rgba(59,130,246,0.3); } }" +
            "  .badge-complete { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }" +
            "  .badge-queued { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }" +
            "  .badge-error { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }" +
            "  .badge-offline { background: rgba(100, 116, 139, 0.2); color: #94a3b8; border: 1px solid #64748b; }" +
            "  .batch-item { display: flex; justify-content: space-between; align-items: center; padding: 11px 0; border-bottom: 1px solid rgba(51, 65, 85, 0.5); }" +
            "  .batch-item:last-child { border-bottom: none; }" +
            "  .batch-left { display: flex; flex-direction: column; }" +
            "  .batch-name { font-size: 13.5px; font-weight: 700; color: #f1f5f9; display: flex; align-items: center; gap: 6px; }" +
            "  .batch-desc { font-size: 11px; color: #64748b; margin-top: 2px; }" +
            "  .batch-timer { font-size: 10.5px; color: #38bdf8; font-family: monospace; margin-top: 2px; }" +
            "  .batch-right { display: flex; align-items: center; gap: 8px; }" +
            "  .btn-mini { padding: 5px 10px; font-size: 10px; font-weight: 700; background: #1e293b; color: #38bdf8; border: 1px solid #38bdf8; border-radius: 6px; cursor: pointer; }" +
            "  .btn-mini:active { background: #38bdf8; color: #000; }" +
            "  .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 6px; }" +
            "  .btn { width: 100%; padding: 12px; border: none; border-radius: 10px; font-size: 13px; font-weight: 700; cursor: pointer; transition: transform 0.1s ease; }" +
            "  .btn:active { transform: scale(0.98); }" +
            "  .btn-primary { background: linear-gradient(135deg, #10b981, #059669); color: #ffffff; box-shadow: 0 4px 12px rgba(16,185,129,0.3); }" +
            "  .btn-purple { background: linear-gradient(135deg, #6366f1, #4f46e5); color: #ffffff; box-shadow: 0 4px 12px rgba(99,102,241,0.3); }" +
            "  .btn-secondary { background: #1a2333; color: #94a3b8; border: 1px solid #334155; margin-top: 8px; font-size: 12px; padding: 10px; }" +
            "  .log-box { background: #030712; border: 1px solid #1f293d; border-radius: 8px; padding: 10px; font-family: monospace; font-size: 10.5px; color: #38bdf8; height: 140px; overflow-y: auto; line-height: 1.4; }" +
            "</style>" +
            "</head><body>" +
            "<div class='header'>" +
            "  <div class='title-wrap'>" +
            "    <div class='title'><span>🔬 Pilot Controller</span></div>" +
            "    <div class='subtitle'>Real-Time Kaggle Cloud GPU Benchmark</div>" +
            "  </div>" +
            "  <div class='live-pill'><div class='pulse-dot'></div><span>LIVE (12s)</span></div>" +
            "</div>" +
            "<div class='card'>" +
            "  <div class='card-title'><span>Active Training Progress</span><span id='pct_text' style='color:#34d399; font-weight:bold;'>0%</span></div>" +
            "  <div class='progress-bg'><div class='progress-fill' id='prog_bar'></div></div>" +
            "  <div style='display:flex; justify-content:space-between; font-size:11px; color:#94a3b8;'>" +
            "    <span id='runs_text'>Manifest: 84 Benchmark Runs</span>" +
            "    <span id='gpus_text' style='color:#60a5fa;'>Checking GPUs...</span>" +
            "  </div>" +
            "</div>" +
            "<div class='card'>" +
            "  <div class='card-title'><span>Kaggle Cloud GPU Batches</span><span style='font-size:10px; color:#64748b;'>21 Runs / Batch</span></div>" +
            "  <div class='batch-item'>" +
            "    <div class='batch-left'>" +
            "      <div class='batch-name'>Batch 0 (Runs 0–20)</div>" +
            "      <div class='batch-desc'>CLEAN &amp; SYM 0.2 (21 runs)</div>" +
            "      <div class='batch-timer' id='b0_timer'></div>" +
            "    </div>" +
            "    <div class='batch-right'><button class='btn-mini' onclick='Android.triggerBatch(0)'>RUN</button><span class='badge badge-offline' id='b0_badge'>POLLING</span></div>" +
            "  </div>" +
            "  <div class='batch-item'>" +
            "    <div class='batch-left'>" +
            "      <div class='batch-name'>Batch 1 (Runs 21–41)</div>" +
            "      <div class='batch-desc'>SYM 0.2 &amp; SYM 0.5 (21 runs)</div>" +
            "      <div class='batch-timer' id='b1_timer'></div>" +
            "    </div>" +
            "    <div class='batch-right'><button class='btn-mini' onclick='Android.triggerBatch(1)'>RUN</button><span class='badge badge-offline' id='b1_badge'>POLLING</span></div>" +
            "  </div>" +
            "  <div class='batch-item'>" +
            "    <div class='batch-left'>" +
            "      <div class='batch-name'>Batch 2 (Runs 42–62)</div>" +
            "      <div class='batch-desc'>SYM 0.5 &amp; ASYM 0.4 (21 runs)</div>" +
            "      <div class='batch-timer' id='b2_timer'></div>" +
            "    </div>" +
            "    <div class='batch-right'><button class='btn-mini' onclick='Android.triggerBatch(2)'>RUN</button><span class='badge badge-offline' id='b2_badge'>POLLING</span></div>" +
            "  </div>" +
            "  <div class='batch-item'>" +
            "    <div class='batch-left'>" +
            "      <div class='batch-name'>Batch 3 (Runs 63–83)</div>" +
            "      <div class='batch-desc'>ASYM 0.4 &amp; Final Seeds (21 runs)</div>" +
            "      <div class='batch-timer' id='b3_timer'></div>" +
            "    </div>" +
            "    <div class='batch-right'><button class='btn-mini' onclick='Android.triggerBatch(3)'>RUN</button><span class='badge badge-offline' id='b3_badge'>POLLING</span></div>" +
            "  </div>" +
            "</div>" +
            "<div class='btn-grid'>" +
            "  <button class='btn btn-primary' onclick='Android.triggerPhase1()'>🚀 RUN PHASE 1 (B0+B1)</button>" +
            "  <button class='btn btn-purple' onclick='Android.triggerPhase2()'>🚀 RUN PHASE 2 (B2+B3)</button>" +
            "</div>" +
            "<div class='btn-grid' style='margin-top:8px;'>" +
            "  <button class='btn btn-secondary' style='margin-top:0;' onclick='Android.refresh()'>🔄 REFRESH STATUS</button>" +
            "  <button class='btn btn-secondary' style='margin-top:0; background:rgba(239,68,68,0.12); border-color:#ef4444; color:#f87171;' onclick='Android.resetAll()'>⚠️ RESET ALL</button>" +
            "</div>" +
            "<div class='card' style='margin-top:10px;'>" +
            "  <div class='card-title'><span>Live Cloud Console</span></div>" +
            "  <div class='log-box' id='log_box'>[SYSTEM] Controller online. Polling Kaggle GPU servers every 4s...</div>" +
            "</div>" +
            "<script>" +
            "  var runStartTimes = {};" +
            "  setInterval(function() {" +
            "    var now = Math.floor(Date.now()/1000);" +
            "    for(var k in runStartTimes) {" +
            "      if(runStartTimes[k] > 0) {" +
            "        var el = document.getElementById(k + '_timer');" +
            "        if(el) {" +
            "          var diff = now - runStartTimes[k];" +
            "          var m = Math.floor(diff/60);" +
            "          var s = diff % 60;" +
            "          el.innerText = '⚡ GPU Active: ' + (m<10?'0':'') + m + 'm ' + (s<10?'0':'') + s + 's';" +
            "        }" +
            "      }" +
            "    }" +
            "  }, 1000);" +
            "  function updateUI(b0, b1, b2, b3, runningCount) {" +
            "    function handleBadge(id, timerId, st) {" +
            "      var el = document.getElementById(id);" +
            "      var tel = document.getElementById(timerId);" +
            "      if(!el) return;" +
            "      el.innerText = st;" +
            "      var prefix = id.split('_')[0];" +
            "      if (st === 'RUNNING') {" +
            "        el.className = 'badge badge-running';" +
            "        if(!runStartTimes[prefix]) runStartTimes[prefix] = Math.floor(Date.now()/1000);" +
            "      } else {" +
            "        runStartTimes[prefix] = 0;" +
            "        if(tel) tel.innerText = '';" +
            "        if (st === 'COMPLETE') el.className = 'badge badge-complete';" +
            "        else if (st === 'QUEUED') el.className = 'badge badge-queued';" +
            "        else if (st === 'ERROR' || st === 'FAILED') el.className = 'badge badge-error';" +
            "        else el.className = 'badge badge-offline';" +
            "      }" +
            "    }" +
            "    handleBadge('b0_badge', 'b0_timer', b0);" +
            "    handleBadge('b1_badge', 'b1_timer', b1);" +
            "    handleBadge('b2_badge', 'b2_timer', b2);" +
            "    handleBadge('b3_badge', 'b3_timer', b3);" +
            "    var activeGpus = runningCount;" +
            "    document.getElementById('gpus_text').innerText = activeGpus > 0 ? (activeGpus + ' / 2 GPUs Active') : 'Cloud GPUs Idle';" +
            "    if(b0==='RUNNING') { document.getElementById('prog_bar').style.width = '25%'; document.getElementById('pct_text').innerText = 'Phase 1 Training'; }" +
            "    else if(b0==='COMPLETE' && b1==='RUNNING') { document.getElementById('prog_bar').style.width = '50%'; document.getElementById('pct_text').innerText = 'Phase 1 (50%)'; }" +
            "    else if(b0==='COMPLETE' && b1==='COMPLETE') { document.getElementById('prog_bar').style.width = '50%'; document.getElementById('pct_text').innerText = 'Phase 1 Done (50%)'; }" +
            "    document.getElementById('runs_text').innerText = 'Phase 1: B0 & B1 | Phase 2: B2 & B3';" +
            "  }" +
            "  function addLog(msg) {" +
            "    var box = document.getElementById('log_box');" +
            "    var now = new Date().toTimeString().split(' ')[0];" +
            "    box.innerHTML = '[' + now + '] ' + msg + '<br>' + box.innerHTML;" +
            "  }" +
            "</script>" +
            "</body></html>";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        webView = findViewById(R.id.webview);
        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);

        webView.addJavascriptInterface(new WebAppInterface(), "Android");
        webView.loadDataWithBaseURL(null, HTML_PAGE, "text/html", "UTF-8", null);

        startAutoPoll();
    }

    public class WebAppInterface {
        @JavascriptInterface
        public void refresh() {
            fetchStatusAsync();
        }

        @JavascriptInterface
        public void triggerBatch(int batchIdx) {
            pushBatchAsync(batchIdx);
        }

        @JavascriptInterface
        public void triggerPhase1() {
            pushBatchAsync(0);
            pushBatchAsync(1);
        }

        @JavascriptInterface
        public void triggerPhase2() {
            pushBatchAsync(2);
            pushBatchAsync(3);
        }

        @JavascriptInterface
        public void resetAll() {
            resetAllBatchesAsync();
        }
    }

    private void startAutoPoll() {
        pollHandler.postDelayed(new Runnable() {
            @Override
            public void run() {
                fetchStatusAsync();
                pollHandler.postDelayed(this, 12000); // Safe 12-second poll interval
            }
        }, 1000);
    }

    private void fetchStatusAsync() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                final String s0 = queryKernelStatus("label-noise-pilot-batch-0");
                try { Thread.sleep(500); } catch (Exception ignored) {}
                final String s1 = queryKernelStatus("label-noise-pilot-batch-1");
                try { Thread.sleep(500); } catch (Exception ignored) {}
                final String s2 = queryKernelStatus("label-noise-pilot-batch-2");
                try { Thread.sleep(500); } catch (Exception ignored) {}
                final String s3 = queryKernelStatus("label-noise-pilot-batch-3");

                int running = 0;
                if ("RUNNING".equals(s0) || "QUEUED".equals(s0)) running++;
                if ("RUNNING".equals(s1) || "QUEUED".equals(s1)) running++;
                if ("RUNNING".equals(s2) || "QUEUED".equals(s2)) running++;
                if ("RUNNING".equals(s3) || "QUEUED".equals(s3)) running++;

                final int finalRun = running;

                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        String js = "updateUI('" + s0 + "', '" + s1 + "', '" + s2 + "', '" + s3 + "', " + finalRun + ");";
                        webView.evaluateJavascript(js, null);
                    }
                });
            }
        });
    }

    private String queryKernelStatus(String slug) {
        try {
            String urlStr = "https://www.kaggle.com/api/v1/kernels/status?userName=" + KAGGLE_USER + "&kernelSlug=" + slug;
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(6000);
            conn.setReadTimeout(6000);

            String auth = KAGGLE_USER + ":" + KAGGLE_KEY;
            String encodedAuth = Base64.encodeToString(auth.getBytes(StandardCharsets.UTF_8), Base64.NO_WRAP);
            conn.setRequestProperty("Authorization", "Basic " + encodedAuth);

            int code = conn.getResponseCode();
            if (code == 200) {
                BufferedReader in = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = in.readLine()) != null) sb.append(line);
                in.close();

                JSONObject obj = new JSONObject(sb.toString());
                String st = obj.optString("status", "QUEUED").toUpperCase();
                if (st.contains("COMPLETE")) return "COMPLETE";
                if (st.contains("RUNNING")) return "RUNNING";
                if (st.contains("QUEUED")) return "QUEUED";
                if (st.contains("ERROR") || st.contains("FAILED") || st.contains("CANCEL")) return "ERROR";
                return st;
            } else if (code == 429) {
                return "SYNCING";
            } else {
                return "OFFLINE";
            }
        } catch (Exception e) {
            return "SYNCING";
        }
    }

    private void pushBatchAsync(final int batchIdx) {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "🚀 Launching Batch " + batchIdx + " to Kaggle GPU...", Toast.LENGTH_SHORT).show();
                        webView.evaluateJavascript("addLog('Launching Batch " + batchIdx + " to Kaggle GPU...');", null);
                    }
                });

                String code = "# Auto-generated batch runner for Kaggle Pilot Execution\n" +
                        "import os\n" +
                        "import sys\n" +
                        "import subprocess\n" +
                        "import tarfile\n" +
                        "import shutil\n\n" +
                        "BATCH_INDEX = " + batchIdx + "\n" +
                        "NUM_BATCHES = 4\n\n" +
                        "print(f'=== STARTING KAGGLE PILOT BATCH {BATCH_INDEX}/{NUM_BATCHES} ===')\n\n" +
                        "# 1. Clone repository\n" +
                        "repo_dir = '/kaggle/working/repo'\n" +
                        "if os.path.exists(repo_dir):\n" +
                        "    shutil.rmtree(repo_dir)\n\n" +
                        "subprocess.run(['git', 'clone', 'https://github.com/VAIBHAV7848/label-noise-generalisation.git', repo_dir], check=True)\n" +
                        "os.chdir(repo_dir)\n\n" +
                        "# 2. Download CIFAR-10\n" +
                        "subprocess.run([sys.executable, 'scripts/download_cifar10.py'], check=True)\n\n" +
                        "# 3. Output directory for this batch\n" +
                        "out_dir = f'/kaggle/working/results_batch_{BATCH_INDEX}'\n" +
                        "os.makedirs(out_dir, exist_ok=True)\n\n" +
                        "# 4. Run pilot batch\n" +
                        "cmd = [\n" +
                        "    sys.executable,\n" +
                        "    '-m',\n" +
                        "    'src.training.run_pilot',\n" +
                        "    '--batch-index', str(BATCH_INDEX),\n" +
                        "    '--num-batches', str(NUM_BATCHES),\n" +
                        "    '--data-dir', 'data',\n" +
                        "    '--output-dir', out_dir,\n" +
                        "    '--device', 'cuda'\n" +
                        "]\n" +
                        "print('Executing command:', ' '.join(cmd))\n" +
                        "subprocess.run(cmd, check=True)\n\n" +
                        "# 5. Archive results to root output directory\n" +
                        "archive_path = f'/kaggle/working/pilot_results_batch_{BATCH_INDEX}.tar.gz'\n" +
                        "with tarfile.open(archive_path, 'w:gz') as tar:\n" +
                        "    tar.add(out_dir, arcname='pilot_results')\n\n" +
                        "# Clean up working files to minimize archive footprint\n" +
                        "shutil.rmtree(repo_dir, ignore_errors=True)\n" +
                        "shutil.rmtree(out_dir, ignore_errors=True)\n\n" +
                        "print(f'=== BATCH {BATCH_INDEX} FINISHED. ARCHIVE SAVED AT {archive_path} ===')\n";

                try {
                    String urlStr = "https://www.kaggle.com/api/v1/kernels/push";
                    URL url = new URL(urlStr);
                    HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                    conn.setRequestMethod("POST");
                    conn.setConnectTimeout(10000);
                    conn.setReadTimeout(10000);
                    conn.setDoOutput(true);
                    conn.setRequestProperty("Content-Type", "application/json");

                    String auth = KAGGLE_USER + ":" + KAGGLE_KEY;
                    String encodedAuth = Base64.encodeToString(auth.getBytes(StandardCharsets.UTF_8), Base64.NO_WRAP);
                    conn.setRequestProperty("Authorization", "Basic " + encodedAuth);

                    JSONObject payload = new JSONObject();
                    payload.put("slug", KAGGLE_USER + "/label-noise-pilot-batch-" + batchIdx);
                    payload.put("newTitle", "Label Noise Pilot Batch " + batchIdx);
                    payload.put("text", code);
                    payload.put("language", "python");
                    payload.put("kernelType", "script");
                    payload.put("isPrivate", true);
                    payload.put("enableGpu", true);
                    payload.put("enableTpu", false);
                    payload.put("enableInternet", true);
                    payload.put("datasetSources", new JSONArray());
                    payload.put("competitionSources", new JSONArray());
                    payload.put("kernelSources", new JSONArray());

                    OutputStream os = conn.getOutputStream();
                    os.write(payload.toString().getBytes(StandardCharsets.UTF_8));
                    os.flush();
                    os.close();

                    int codeResp = conn.getResponseCode();
                    BufferedReader in = new BufferedReader(new InputStreamReader(
                            codeResp >= 200 && codeResp < 300 ? conn.getInputStream() : conn.getErrorStream()
                    ));
                    StringBuilder sb = new StringBuilder();
                    String line;
                    while ((line = in.readLine()) != null) sb.append(line);
                    in.close();

                    final String resText = sb.toString();
                    mainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            if (resText.contains("error") && !resText.contains("\"hasError\":false")) {
                                webView.evaluateJavascript("addLog('⚠️ Batch " + batchIdx + " Notice: " + resText.replace("'", "") + "');", null);
                            } else {
                                webView.evaluateJavascript("addLog('✅ Batch " + batchIdx + " launched to GPU!');", null);
                            }
                        }
                    });

                } catch (final Exception e) {
                    mainHandler.post(new Runnable() {
                        @Override
                        public void run() {
                            webView.evaluateJavascript("addLog('❌ Push Error for Batch " + batchIdx + ": " + e.getMessage() + "');", null);
                        }
                    });
                }

                fetchStatusAsync();
            }
        });
    }

    private void resetAllBatchesAsync() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                mainHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        Toast.makeText(MainActivity.this, "🔄 Resetting all batches on Kaggle...", Toast.LENGTH_SHORT).show();
                        webView.evaluateJavascript("addLog('🔄 Resetting all 4 batches on Kaggle...');", null);
                    }
                });

                for (int b = 0; b < 4; b++) {
                    try {
                        String urlStr = "https://www.kaggle.com/api/v1/kernels/push";
                        URL url = new URL(urlStr);
                        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                        conn.setRequestMethod("POST");
                        conn.setConnectTimeout(8000);
                        conn.setReadTimeout(8000);
                        conn.setDoOutput(true);
                        conn.setRequestProperty("Content-Type", "application/json");

                        String auth = KAGGLE_USER + ":" + KAGGLE_KEY;
                        String encodedAuth = Base64.encodeToString(auth.getBytes(StandardCharsets.UTF_8), Base64.NO_WRAP);
                        conn.setRequestProperty("Authorization", "Basic " + encodedAuth);

                        JSONObject payload = new JSONObject();
                        payload.put("slug", KAGGLE_USER + "/label-noise-pilot-batch-" + b);
                        payload.put("newTitle", "Label Noise Pilot Batch " + b);
                        payload.put("text", "print('Fresh Reset: Ready for pilot execution...')");
                        payload.put("language", "python");
                        payload.put("kernelType", "script");
                        payload.put("isPrivate", true);
                        payload.put("enableGpu", false);
                        payload.put("enableTpu", false);
                        payload.put("enableInternet", false);
                        payload.put("datasetSources", new JSONArray());
                        payload.put("competitionSources", new JSONArray());
                        payload.put("kernelSources", new JSONArray());

                        OutputStream os = conn.getOutputStream();
                        os.write(payload.toString().getBytes(StandardCharsets.UTF_8));
                        os.flush();
                        os.close();

                        int codeResp = conn.getResponseCode();
                        final int batchNum = b;
                        mainHandler.post(new Runnable() {
                            @Override
                            public void run() {
                                webView.evaluateJavascript("addLog('Reset Batch " + batchNum + " -> READY');", null);
                            }
                        });
                    } catch (Exception ignored) {}
                }

                fetchStatusAsync();
            }
        });
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdown();
        pollHandler.removeCallbacksAndMessages(null);
    }
}
