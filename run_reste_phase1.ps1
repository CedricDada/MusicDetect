# =====================================================================
# Reprise de la phase 1 -- relancable sans risque.
#
#   .\run_reste_phase1.ps1
#
# Le script saute tout ce qui est deja termine : il lit les modeles
# presents sur le disque et la colonne "test" des fichiers de metriques.
# S'il est interrompu, il suffit de le relancer, il reprend au bon
# endroit. Tout est journalise dans logs\reste_<horodatage>.log, sortie
# de uv incluse (Tee-Object, et non Start-Transcript qui perd la sortie
# des executables natifs sous PowerShell 5.1).
#
# AVANT DE LANCER, voir la section "veille et mises a jour" du message
# qui accompagne ce script : une fenetre tuee par un redemarrage de
# Windows Update fait perdre l'entrainement en cours.
# =====================================================================

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root
$cfg = "configs/phase1.yaml"

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logDir = Join-Path $root "logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }
$script:log = Join-Path $logDir "reste_$stamp.log"
New-Item -ItemType File $script:log | Out-Null

$global:failures = @()
$t0 = Get-Date

$gens = @("fmc:musicgen", "fmc:stable_audio_open", "fmc:audioldm2", "fmc:musicldm", "fmc:mustango")

# Ecrit a la fois a l'ecran et dans le journal.
function Write-Log {
  param([string]$Msg = "")
  Write-Host $Msg
  Add-Content -Path $script:log -Value $Msg -Encoding UTF8
}

# Liste des tests pour un codec donne. Le controle en-domaine du codec
# lui-meme est ajoute quand il differe de 6 kbps : c'est le controle qui
# manquait aux evaluations deja faites sur les modeles 12 kbps.
function Get-Tests {
  param([string]$Codec)
  $t = @("fma:encodec_24k_6kbps", "fma:identity_24k")
  if ($Codec -ne "encodec_24k_6kbps") { $t += "fma:$Codec" }
  return $t + $gens
}

function Get-TestArgs {
  param([string[]]$Tests)
  $a = @()
  foreach ($t in $Tests) { $a += "--test"; $a += $t }
  return $a
}

# Une evaluation est consideree faite si son fichier de metriques
# contient deja une ligne pour chacun des tests demandes.
function Test-EvalComplete {
  param([string]$Csv, [string[]]$Tests)
  if (-not (Test-Path $Csv)) { return $false }
  try { $rows = @(Import-Csv $Csv) } catch { return $false }
  $have = @($rows | ForEach-Object { $_.test })
  foreach ($t in $Tests) { if ($have -notcontains $t) { return $false } }
  return $true
}

# La sortie de uv (stdout et stderr, barres de progression comprises)
# passe par Tee-Object pour atterrir dans le journal. On teste
# $LASTEXITCODE et non $? : sous PowerShell 5.1, 2>&1 sur un executable
# natif met $? a false meme quand le code de retour est 0.
function Invoke-Step {
  param([string]$Label, [scriptblock]$Body)
  Write-Log ""
  Write-Log "=== RUN   $Label        $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
  $global:LASTEXITCODE = 0
  & $Body 2>&1 | Tee-Object -FilePath $script:log -Append
  if ($LASTEXITCODE -ne 0) {
    Write-Log "!!! ECHEC $Label  (code $LASTEXITCODE) -- on continue"
    $global:failures += $Label
  } else {
    Write-Log "--- OK    $Label        $(Get-Date -Format 'HH:mm:ss')"
  }
}

Write-Log "Journal : $script:log"
Write-Log "Depart  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "PID     : $PID"

# ---------------------------------------------------------------------
# 1. Boucle M2/M3 : 3 graines x 2 codecs x 2 detecteurs
# ---------------------------------------------------------------------
Write-Log ""
Write-Log "########## 1. Entrainements et evaluations M2/M3 ##########"

foreach ($s in 1, 2, 3) {
  $tag = "phase1_s$s"
  foreach ($c in @("encodec_24k_6kbps", "encodec_24k_12kbps")) {
    foreach ($d in @("m2", "m3")) {

      $model = "data\work\models\$tag\$($d)__$c.pt"
      if (Test-Path $model) {
        Write-Log "[SAUTE]  train    $d / $c / $tag"
      } else {
        # --jobs parallelise l'augmentation de vitesse de M3 (joblib). Son defaut est 1,
        # ce qui explique les 1 h 39 de la premiere augmentation.
        Invoke-Step "train    $d / $c / $tag" {
          uv run musicdetect train --detector $d --codec $c --set "$d.seed=$s" -c $cfg --tag $tag --jobs 4
        }
      }

      $tests = Get-Tests $c
      $mcsv = "results\$tag\$($d)__$c.csv"
      if (Test-EvalComplete $mcsv $tests) {
        Write-Log "[SAUTE]  evaluate $d / $c / $tag"
      } else {
        $ta = Get-TestArgs $tests
        Invoke-Step "evaluate $d / $c / $tag" {
          uv run musicdetect evaluate --detector $d --codec $c @ta -c $cfg --tag $tag --jobs 4
        }
      }
    }
  }
}

# ---------------------------------------------------------------------
# 2. M0 et M1 : controle en-domaine 12 kbps qui manquait
# ---------------------------------------------------------------------
Write-Log ""
Write-Log "########## 2. Controle en-domaine 12 kbps (M0, M1) ##########"

foreach ($c in @("encodec_24k_6kbps", "encodec_24k_12kbps")) {
  $tests = Get-Tests $c
  $ta = Get-TestArgs $tests

  $mcsv = "results\phase1\m0-residual__$c.csv"
  if (Test-EvalComplete $mcsv $tests) {
    Write-Log "[SAUTE]  evaluate m0-residual / $c / phase1"
  } else {
    Invoke-Step "evaluate m0-residual / $c / phase1" {
      uv run musicdetect evaluate --detector m0 --variant residual --codec $c @ta -c $cfg --tag phase1 --jobs 4
    }
  }

  $mcsv = "results\phase1\m1__$c.csv"
  if (Test-EvalComplete $mcsv $tests) {
    Write-Log "[SAUTE]  evaluate m1 / $c / phase1"
  } else {
    Invoke-Step "evaluate m1 / $c / phase1" {
      uv run musicdetect evaluate --detector m1 --codec $c @ta -c $cfg --tag phase1 --jobs 4
    }
  }
}

# ---------------------------------------------------------------------
# 3. Attaques (H4). On passe la liste complete des tests et on cible
#    les attaques avec --attack-on, sinon le fichier de metriques du
#    modele serait reecrit avec une seule ligne.
# ---------------------------------------------------------------------
Write-Log ""
Write-Log "########## 3. Grille d'attaques ##########"

$taSix = Get-TestArgs (Get-Tests "encodec_24k_6kbps")

$att = "results\phase1\attacks_m1__encodec_24k_6kbps.csv"
if (Test-Path $att) {
  Write-Log "[SAUTE]  attaques m1 / phase1"
} else {
  Invoke-Step "attaques m1 / encodec_24k_6kbps / phase1" {
    uv run musicdetect evaluate --detector m1 --codec encodec_24k_6kbps @taSix --attacks --attack-on fmc:musicgen -c $cfg --tag phase1 --jobs 4
  }
}

foreach ($d in @("m3", "m2")) {
  $att = "results\phase1_s1\attacks_$($d)__encodec_24k_6kbps.csv"
  if (Test-Path $att) {
    Write-Log "[SAUTE]  attaques $d / phase1_s1"
  } else {
    Invoke-Step "attaques $d / encodec_24k_6kbps / phase1_s1" {
      uv run musicdetect evaluate --detector $d --codec encodec_24k_6kbps @taSix --attacks --attack-on fmc:musicgen -c $cfg --tag phase1_s1 --jobs 4
    }
  }
}

# ---------------------------------------------------------------------
# 4. Rapports -- toujours regeneres, ils agregent ce qui precede
# ---------------------------------------------------------------------
Write-Log ""
Write-Log "########## 4. Rapports ##########"

foreach ($tag in @("phase1", "phase1_s1", "phase1_s2", "phase1_s3")) {
  Invoke-Step "report $tag" { uv run musicdetect report -c $cfg --tag $tag }
}

# ---------------------------------------------------------------------
# 5. Comparaisons appariees M1 contre M0 (H3)
# ---------------------------------------------------------------------
Write-Log ""
Write-Log "########## 5. Comparaisons appariees M1 vs M0 ##########"

$sc = "results\phase1\scores"
$cmp = "results\phase1\compare_m1_vs_m0.txt"
if (Test-Path $cmp) { Remove-Item $cmp }

foreach ($g in @("musicgen", "stable_audio_open", "audioldm2", "musicldm", "mustango")) {
  $a = "$sc\m1__encodec_24k_6kbps__fmc_$g.csv"
  $b = "$sc\m0-residual__encodec_24k_6kbps__fmc_$g.csv"
  if ((Test-Path $a) -and (Test-Path $b)) {
    "===== $g =====" | Out-File -Append -Encoding utf8 $cmp
    Invoke-Step "compare m1 vs m0 sur $g" {
      uv run musicdetect compare $a $b --bootstrap 1000 | Out-File -Append -Encoding utf8 $cmp
    }
  } else {
    Write-Log "[SAUTE]  compare $g : fichier de scores absent"
  }
}
if (Test-Path $cmp) {
  Write-Log ""
  Get-Content $cmp | ForEach-Object { Write-Log $_ }
}

# ---------------------------------------------------------------------
Write-Log ""
Write-Log "##########################################################"
Write-Log "Termine  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "Duree    : $([math]::Round(((Get-Date) - $t0).TotalMinutes, 1)) minutes"
if ($global:failures.Count -eq 0) {
  Write-Log "Echecs   : aucun"
} else {
  Write-Log "Echecs   : $($global:failures.Count)"
  foreach ($f in $global:failures) { Write-Log "  - $f" }
  Write-Log "Relance le script : il reprendra uniquement ce qui manque."
}
Write-Log "Journal  : $script:log"
Write-Log "##########################################################"
