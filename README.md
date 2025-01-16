# CVE Reference Link Crawler

A Python-based tool to extract, download, and process CVE reference content from the National Vulnerability Database (NVD).

This tool focuses on specific CVEs from a target list, downloads their reference content, converts it to a standardized text format, and extracts structured vulnerability information using LLM analysis.

For example, CVE-2021-4034 has 11 unique links that contain vulnerability information which this tool can process and analyze.

## Example
[CVE-2021-4034](https://nvd.nist.gov/vuln/detail/cve-2021-4034) has 11 unique links including one from the CNA: https://access.redhat.com/security/vulnerabilities/RHSB-2022-001

The links are in, e.g. for CVE-2021-4034:
- "References to Advisories, Solutions, and Tools" on https://nvd.nist.gov/vuln/detail/CVE-2021-4034
- "references" section of nvd.json for CVE-2021-4034
```json
   "cve": {
      "id": "CVE-2021-4034",
      .....
      "references": [
        {
          "url": "..."
````

## Problem Space

A significant number of CVE Descriptions do not contain all the necessary vulnerability information and it is instead available in the reference links for that CVE.

However, this data is
- often no longer present i.e. dead links
- any format: from text, html, md, pdf, website,....
- unstructured
- multi-format i.e. can contain images, animated images, text,
- single or multi CVE i.e. the link can include details for many CVEs, not just the one of interest
- in different spoken languages

It would be good to have this reference content available:
1. persistently (to avoid link rot, or moved content)
2. in a single format i.e. as text only that can be read by people or machines
3. in a single language e.g. English.
4. per single CVE
5. with only the vulnerability information extracted (not page headers, footers, other CVE info,....)

### Link Rot

[Research by Jerry Gamblin](https://gist.github.com/jgamblin/93c10ce6ebc7a688d60eb2b21f8216b3) shows that (as at mid 2024) there are:
- 909,391 CVE reference links 
- ~~13% of all CVE reference links are dead.
   - **this is a best case %** as this just refers to the associated domain names that no longer exist. 
   - In other cases, the domain may be active, but the content no longer exists, or no longer exists in the original location or version of the site.

### Generic Links

Some CVE reference links are not specific e.g. https://www.forescout.com/blog/ per https://nvd.nist.gov/vuln/detail/cve-2022-31207 so the content is lost over time as new content is added.

### NVD CVE Enrichment
> Enrichment efforts begin with reviewing any reference material provided with the CVE record and assigns appropriate reference tags. This helps organize the various data sources to help researchers find the relevant information for their needs. Enrichment efforts also include manual searches of the internet to ensure that any other available and relevant information is used for the enrichment process. NVD enrichment efforts only use publicly available materials in the enrichment process.


### Overview

This tool:
1. Loads target CVEs from a CSV file
2. Processes only the specified CVEs from the NVD JSON data
3. Downloads and archives reference content for these CVEs
4. Converts various file formats to text using MarkItDown
5. Creates a structured archive of both raw and processed content
6. Uses LLM analysis to extract structured vulnerability information

## Content Processing Workflow

### Three-Phase Processing
The tool processes content in three phases to ensure comprehensive coverage:

1. Initial Phase:
   - Processes all direct URLs from the CVE data
   - Creates CVE directories with raw and text content

2. Secondary Phase:
   - Scans generated text files for additional CVE-specific URLs
   - Downloads and processes these secondary URLs
   - Maintains the same directory structure for consistency

3. Analysis Phase:
   - Uses LLM to analyze all collected text content
   - Extracts structured vulnerability information including:
     - Root cause of vulnerability
     - Weaknesses/vulnerabilities present
     - Impact of exploitation
     - Attack vectors
     - Required attacker capabilities/position
   - Translates non-English content to English automatically
   - Stores results in refined/ directory

>[!NOTE]
> This Secondary Phase is particularly important because relevant vulnerability information is often not in the directly linked document. For example:
> - https://nvd.nist.gov/vuln/detail/CVE-2021-0955 links to
>   - https://source.android.com/docs/security/bulletin/2021-12-01 (a bulletin with multiple CVEs) Which links to the actual relevant content: 
>      - https://android.googlesource.com/platform/packages/providers/MediaProvider/+/e81d03db8006fddf6e7c8a8eda1b73743314a214

### Vulnerability Information Extraction

The tool uses Google's Gemini API to analyze and extract vulnerability information:

1. Configuration:
   - Uses environment variables for API keys
   - Configurable safety settings to allow security content
   - Customizable generation parameters

2. Processing:
   - Combines all text content from references
   - Verifies content relevance to specific CVE
   - Extracts structured vulnerability information
   - Handles multi-language content with translation

3. Output:
   - Creates refined.txt with structured analysis
   - Includes original technical details
   - Removes unrelated content
   - Notes additional details beyond CVE description

### Example Output Structure

For each CVE, the tool creates:
```
data_out/
└── CVE-YYYY-XXXXX/
    ├── links.txt              # URLs from CVE references
    ├── raw/                   # Original content
    │   ├── site1_hash_timestamp.html
    │   └── site2_hash_timestamp.pdf
    ├── text/                  # Converted content
    │   ├── site1_hash_timestamp.md
    │   └── site2_hash_timestamp.md
    └── refined/               # Analyzed content
        ├── combined.md       # All reference content
        └── refined.md        # Extracted vulnerability info
````



### Content Conversion
The tool uses multiple methods to convert content to readable text:

1. Primary Method: MarkItDown
   - Handles multiple formats: PDF, Images (with OCR), HTML, CSV, JSON, XML
   - Provides consistent output format

2. Fallback Methods:
   - PyPDF2 for PDFs that fail with MarkItDown
   - Custom handlers for specific sources

### URL Handlers

The tool includes specialized handlers for specific sources to ensure proper content extraction:

1. Google Source Handler
   Some CVE References are (indirect) links to Google Android e.g. 
   
   CVE-2021-0955 
   - links to https://source.android.com/security/bulletin/2021-12-01
     - links to https://android.googlesource.com/platform/packages/providers/MediaProvider/+/e81d03db8006fddf6e7c8a8eda1b73743314a214

   These need to be changed get get the content:
   
   - Transforms URLs for raw content access
   - Decodes base64-encoded responses
   - Formats commit information:
     ```
     Original: .../MediaProvider/+/e81d03db8006fddf6e7c8a8eda1b73743314a214
     Handled: .../MediaProvider/+/e81d03db8006fddf6e7c8a8eda1b73743214?format=TEXT
     ```

2. CISA URL Handler
   
   Some CVE References are links to CISA e.g. https://nvd.nist.gov/vuln/detail/CVE-2022-31204

   The tool handles the changes to the CISA links:

   - Handles post-February 2023 website reorganization
   - Automatically redirects old URLs:
     ```
     Old: www.cisa.gov/uscert/ics/advisories/icsa-22-179-02
     New: www.cisa.gov/news-events/ics-advisories/icsa-22-179-02
     ```

3. YouTube Transcript Handler
   
   Some (~40) CVE References are links to YouTube e.g. https://nvd.nist.gov/vuln/detail/CVE-2020-15912. 
   - Only a subset of these have transcripts.

   The YouTube transcript will be downloaded if available.
   - Note: YouTube transcripts are only available if the video has closed captions enabled by the creator or auto-generated by YouTube.

   - Extracts closed captions/transcripts from YouTube videos
   - Supports multiple URL formats:
     ```
     youtube.com/watch?v=VIDEO_ID
     youtu.be/VIDEO_ID
     youtube.com/v/VIDEO_ID
     youtube.com/embed/VIDEO_ID
     ```
   - Formats transcripts with timestamps:
     ```
     [00:00] Welcome to the video
     [00:05] Today we'll discuss...
     ```
   - Requires installation of transcript API:
     ```
     pip install youtube-transcript-api
     ```
Each handler automatically processes URLs from its respective source, converting content into a consistent text format for analysis. The handlers integrate seamlessly with the crawler's core functionality, requiring no additional configuration from users.



### URL Filtering
The tool maintains multiple filtering mechanisms:

1. Dead Domains:
   - Loaded from data_in/dead_domains.csv if present
   - Format: CSV with Domain,count columns
   - Example entries:
     ```csv
     Domain,count
     www.securityfocus.com,82853
     osvdb.org,11271
     ```

2. robots.txt Compliance:
   - Respects robots.txt directives
   - Uses provided User-Agent for checking permissions
   - Caches robots.txt content for 1 hour to reduce requests
   - Falls back to allowing access if robots.txt cannot be fetched

3. Configurable Ignore List:
   - Known problematic domains
   - Circular references (e.g., www.cve.org)
   - Sources requiring authentication

Configure ignored URLs in config.py:
```python
IGNORED_URLS = [
    "www.cve.org",
    "cve.org",
    "first.org",
    "nist.gov",
    "www.securitytracker.com",
    "securitytracker.com",
    "exchange.xforce.ibmcloud.com"
]
```

The tool applies these filters in sequence:
- Checks against ignored URLs list
- Verifies domain is not in dead_domains.csv
- Checks robots.txt permissions before crawling



## Project Structure

```
project_root/
├── cve_reference_link_crawler/
│   ├── src/
│   │   ├── config.py          # Configuration settings
│   │   ├── main.py            # Main entry point
│   │   └── cve_ref_crawler/   
│   │       ├── handlers/      # URL handlers
│   │       │   ├── googlesource.py
│   │       │   └── cisa.py
│   │       └── utils/         # Utility functions
│   ├── logs/                  # Processing logs
│   ├── data_in/              
│   │   └── nvd.jsonl          # NVD JSON data
│   └── data_out/              # Output directory
│       └── CVE-YYYY-XXXXX/    # Per-CVE directories
└── cwe_top25/                 # External CVE list
    └── data_in/
        └── top25-mitre-mapping-analysis-2023-public.csv 
```



### Example: CVE-2021-3675

This example demonstrates typical processing outcomes:

1. Initial Links (from links.txt):
   ```
   https://support.hp.com/us-en/document/ish_6411153-6411191-16/hpsbhf03797
   https://support.lenovo.com/us/en/product_security/LEN-68054
   https://synaptics.com/.../fingerprint-driver-SGX-security-brief-2022-06-14.pdf
   ```

2. Processing Results:
   - First two links failed (common for vendor-specific pages)
   - PDF successfully processed and converted to text
   - Demonstrates both successful and failed crawling scenarios

## Prerequisites

1. Python 3.8+
2. Git
3. NVD data (from http://nvd.handsonhacking.org/nvd.jsonl)
4. MarkItDown (Microsoft's document conversion utility)
5. Google Cloud API key for Gemini

## Installation

1. Setup Repository:
   ```bash
   # Clone repositories
   git clone https://github.com/CyberSecAI/cwe_top25
   git clone [this-repository-url]
   cd [this-repository-url]

   # Create required directories
   mkdir -p data_in data_out logs
   ```

2. Create Python Environment:
   ```bash
   python -m venv env
   source env/bin/activate  # Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure:
   - Copy NVD data to data_in/nvd.jsonl
   - Create .env file with Google API key
   - Review and modify config.py settings

## Environment Setup

Create a .env file in the env/ directory:
```
GOOGLE_API_KEY=your_api_key_here
```

## Configuration

config.py settings:
```python
# Directories
DATA_IN_DIR = Path("data_in")
DATA_OUT_DIR = Path("data_out")
LOG_DIR = Path("logs")

# Crawler behavior
CRAWLER_SETTINGS = {
    "timeout": 30,
    "retry_count": 3,
    "delay": 1,  # seconds between requests
}

# Logging
LOG_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(levelname)s - %(message)s"
}


# Dead domains and robots.txt
- Place dead_domains.csv in data_in/ directory (optional)
- robots.txt caching duration can be modified in code (default 1 hour)
- User-Agent used for robots.txt checks is configured in CRAWLER_SETTINGS:
```python
CRAWLER_SETTINGS = {
    "headers": {
        "User-Agent": "Mozilla/5.0 ...",
        # other headers...
    }
}
```

## Usage

```
python src/main.py
```

## Error Handling

1. Logs Directory:
   - Located in data_out/logs/
   - Contains timestamped log files
   - Includes both success and error information

2. Common Errors:
   - Authentication Required (e.g., HackerOne)
   - JavaScript-dependent content (e.g., Dell advisories)
   - Rate limiting (e.g., Qualcomm bulletins)
   - Domain-specific issues (e.g., Secomea content)

3. Error Resolution:
   - Check logs/crawler_YYYYMMDD_HHMMSS.log
   - Consider adding problematic domains to IGNORED_URLS
   - Review timeout and retry settings



 
## Refined CVE Reference Link Vulnerability Info


### Interpretation of Code in Links

https://nvd.nist.gov/vuln/detail/CVE-2024-56622

The LLM can add explanation and context to code in references.

>In the Linux kernel, the following vulnerability has been resolved: scsi: ufs: core: sysfs: Prevent div by zero Prevent a division by 0 when monitoring is not enabled.

https://nvd.nist.gov/vuln/detail/CVE-2024-56622 contains these direct links which contain code (patches):

````
https://git.kernel.org/stable/c/0069928727c2e95ca26c738fbe6e4b241aeaaf08
https://git.kernel.org/stable/c/7b21233e5f72d10f08310689f993c1dbdfde9f2c
https://git.kernel.org/stable/c/87bf3ea841a5d77beae6bb85af36b2b3848407ee
https://git.kernel.org/stable/c/9c191055c7abea4912fdb83cb9b261732b25a0c8
https://git.kernel.org/stable/c/eb48e9fc0028bed94a40a9352d065909f19e333c
````
./data_out/CVE-2024-56622/refined/refined.md
````
Based on the provided content, here's an analysis of the vulnerability:

**Root Cause:**
The vulnerability stems from a division-by-zero error within the UFS (Universal Flash Storage) subsystem of the Linux kernel. Specifically, when monitoring is disabled or no requests have been made, the code attempts to calculate the average request latency by dividing the sum of latencies by the number of requests. If no requests have been made, the number of requests will be zero, leading to division-by-zero.

**Weaknesses/Vulnerabilities Present:**
- **Division by Zero:** The code lacks a check to ensure that the divisor (number of requests) is not zero before performing the division. This can cause the kernel to crash.

**Impact of Exploitation:**
- **Kernel Crash:**  Exploiting this vulnerability can lead to a denial-of-service (DoS) by crashing the kernel.

**Attack Vectors:**
- The vulnerability is triggered when reading the `read_req_latency_avg` or `write_req_latency_avg` sysfs attributes associated with the UFS HBA (Host Bus Adapter) when no read or write requests have been made since the UFS driver was initialized, or if monitoring is disabled.

**Required Attacker Capabilities/Position:**
- An attacker needs the capability to read sysfs attributes, which usually requires local access to the system with sufficient permissions (e.g., being root or having necessary read permissions to those files).

**Additional Notes:**
- The fix addresses the vulnerability by adding a check for zero requests before performing the division. If the number of requests is zero, the sysfs entry will emit a '0' instead of performing the division.
- The vulnerability is present in multiple files of the UFS subsystem, and was fixed in the following files:
  - `drivers/ufs/core/ufs-sysfs.c`
  - `drivers/scsi/ufs/ufs-sysfs.c
````

### Secondary Link Info
https://nvd.nist.gov/vuln/detail/cve-2021-26401

>LFENCE/JMP (mitigation V2-2) may not sufficiently mitigate CVE-2017-5715 on some AMD CPUs.

This Description does not contain much useful information and it is not possible to assign a CWE per NIST:

````
CWE-ID            CWE Name	Source
NVD-CWE-noinfo    Insufficient Information
````

./data_out/CVE-2021-26401/refined/refined.md 
````
Based on the provided content, here's an analysis of CVE-2021-26401:

**Root Cause of Vulnerability:**

*   The vulnerability stems from an incomplete mitigation for CVE-2017-5715 (Spectre Variant 2) on some AMD CPUs. Specifically, the `lfence/jmp` mitigation (MITIGATION V2-2) is insufficient. The speculative execution window remains large enough for exploitation.

**Weaknesses/Vulnerabilities Present:**

*   **Insufficient Mitigation:** The `lfence/jmp` mitigation, intended to prevent speculative execution beyond indirect branches, fails to fully protect against Spectre-v2 on certain AMD processors.
*   **Speculative Execution:** The vulnerability relies on the speculative execution capabilities of the CPU, where it predicts branch targets and executes instructions speculatively, potentially leaking information.

**Impact of Exploitation:**

*   **Loss of Confidentiality:** An attacker could potentially infer the contents of arbitrary memory, including memory belonging to other processes or virtual machines. This could lead to the disclosure of sensitive information.

**Attack Vectors:**

*   The attack vector involves exploiting the CPU's speculative execution behavior, bypassing the insufficient `lfence/jmp` mitigation.
*   The attacker would need to craft a sequence of instructions that would trigger the CPU to speculatively execute down the wrong path and leak data through a side channel.

**Required Attacker Capabilities/Position:**

*   The attacker would need to execute code on the vulnerable system. This could be achieved through local access or by exploiting a remote vulnerability.
*   The attacker would need a deep understanding of the CPU architecture and speculative execution mechanisms to successfully craft a working exploit.

**Additional Details from the Content:**

*   **Affected Products:** The vulnerability affects a wide range of AMD processors, including 1st/2nd/3rd Gen EPYC processors, Ryzen desktop, mobile, and HEDT processors, and Threadripper PRO workstation processors.
*   **Mitigation:** AMD recommends using alternative mitigations for CVE-2017-5715, specifically `generic retpoline` (V2-1) or `IBRS` (V2-4).  The Xen Security Advisory also suggests using IBRS on capable AMD hardware, and CET Indirect Branch Tracking on capable Intel hardware for a CET Shadow Stack compatible mitigation.
*   **Xen's perspective:** Xen is affected due to the speculative execution issues. It prefers retpoline for non CET-SS and IBRS with CET-SS, and is working to address issues in its own code base.
*   **Reporting:** The vulnerability was reported by researchers from Intel IPAS STORM.

**Summary of differences from CVE description:**
The provided content gives more detail than what is normally in a CVE description, including:
* specific CPU models affected
* the recommended mitigations (retpoline and IBRS)
* information related to how Xen is affected and its mitigation strategy
* the researchers who reported the issue
````



### Secondary Links

https://nvd.nist.gov/vuln/detail/CVE-2022-20148 

CVE Description
>In TBD of TBD, there is a possible use-after-free due to a race condition. This could lead to local escalation of privilege in the kernel with System execution privileges needed. User interaction is not needed for exploitation.Product: AndroidVersions: Android kernelAndroid ID: A-219513976References: Upstream kernel

The CVE Reference Link is "Pixel Update Bulletin—June 2022" (a collection of links for many CVEs): https://source.android.com/security/bulletin/pixel/2022-06-01
- the bulletin for CVE-2022-20148 links to https://android.googlesource.com/kernel/common/+/528611246fcbd which contains this information


````
f2fs: allow to change discard policy based on cached discard cmds

With the default DPOLICY_BG discard thread is ioaware, which prevents
the discard thread from issuing the discard commands. On low RAM setups,
it is observed that these discard commands in the cache are consuming
high memory. This patch aims to relax the memory pressure on the system
due to f2fs pending discard cmds by changing the policy to DPOLICY_FORCE
based on the nm_i->ram_thresh configured.

Signed-off-by: Sahitya Tummala <stummala@codeaurora.org>
Reviewed-by: Chao Yu <yuchao0@huawei.com>
Signed-off-by: Jaegeuk Kim <jaegeuk@kernel.org>
````

data_out/CVE-2022-20148/refined/refined.md

The data that was autoextracted by the crawler from the reference links is data_out/CVE-2022-20148/refined/refined.md:

````
Based on the provided information, here's an analysis of CVE-2022-20148:

**Root cause of vulnerability:**

The root cause lies in the f2fs (Flash-Friendly File System) discard mechanism. The default discard policy, `DPOLICY_BG`, uses an I/O aware background thread, which can be blocked from issuing discard commands. When dealing with low RAM situations, cached discard commands may consume large amounts of memory, creating memory pressure.

**Weaknesses/vulnerabilities present:**

*   **Memory exhaustion:**  The caching of discard commands in f2fs can lead to high memory consumption on low RAM devices, potentially causing a denial-of-service.
*   **Inefficient discard processing:** The `DPOLICY_BG` thread may not be able to issue discard commands effectively in low RAM scenarios.

**Impact of exploitation:**

*   **Denial of Service (DoS):**  A device with limited memory may become unresponsive or crash due to memory exhaustion caused by the cached discard commands.
*   **Reduced performance:**  The system may experience overall slowdowns due to excessive memory pressure.

**Attack vectors:**

*   **Normal File System Operations:** An attacker doesn't need direct access to the file system. Normal file system operations that lead to generating discard commands will cause the vulnerability to be triggered if the system has low memory.
*   **Low Memory Conditions:** The vulnerability is exacerbated when the system is under memory pressure, making it more likely that a large amount of discard commands is cached.

**Required attacker capabilities/position:**

*   **Ability to cause discard commands**: An attacker only needs to perform file system operations that generate discard requests. No root or elevated privileges is required.
*   **Low RAM conditions:** For the vulnerability to be triggered, low memory conditions need to be present.

**Mitigation:**

The fix involves allowing the discard policy to be changed to `DPOLICY_FORCE` based on the configured `nm_i->ram_thresh`. This allows discard commands to be issued more aggressively in low-memory situations, reducing memory pressure.

**Additional Notes**

*   The provided commit messages are all identical, which implies that the fix involves changing the discard policy under certain conditions.
*   The commit message explicitly mentions "low RAM setups".
*   The fix is related to the Linux Kernel f2fs implementation.
*   The security bulletin for Pixel devices confirms the vulnerability as an Elevation of Privilege (EoP) issue within the Kernel component.
````



### Translation

https://nvd.nist.gov/vuln/detail/cve-2022-30707 contains several reference links, 2 of which are in Japanese:
- https://web-material3.yokogawa.com/1/32780/files/YSAR-22-0006-E.pdf
- https://web-material3.yokogawa.com/19/32780/files/YSAR-22-0006-J.pdf

This text is extracted as with all other reference content, **and then translated to English by an LLM**.

**Japanese**
```markdown
脆弱性詳細:
攻撃者が何らかの方法で同製品がインストールされたコンピューターに侵入できた場合、当該コンピュータ
ーに格納されているアカウント、パスワードを用いて、別の CAMS for HIS が管理するデータが漏洩／改ざ
んされる可能性があります。また、同アカウント、パスワードを用いて、別の CAMS for HIS に不要なファ
イルを作成するリソース枯渇攻撃がおこなわれ、最終的に CAMS for HIS の機能を停止させられる可能性が
あります。

•セキュリティ設計の原則に反した設計(CWE-657)
CVE: CVE-2022-30707
CVSS v3 基本値:6.4
CVSS:3.0/AV:A/AC:H/PR:N/UI:N/S:U/C:L/I:L/A:H
````

````
Based on the provided information, here's an analysis of CVE-2022-30707:

**Root Cause of Vulnerability:**
The vulnerability stems from a flaw in the communication design of Yokogawa's CAMS for HIS (Consolidation Alarm Management Software for Human Interface Station). This is categorized as a violation of secure design principles. Specifically, the system allows for the potential misuse of stored credentials.

**Weaknesses/Vulnerabilities Present:**
- **Violation of Secure Design Principles (CWE-657):** The core weakness lies in how CAMS for HIS handles credentials. When a system is compromised, the stored credentials can be used to access and manipulate data on other CAMS for HIS systems.
- **Insecure Credential Handling:** The vulnerability arises from the fact that credentials stored on one system can be used to authenticate against other systems, leading to unauthorized access.

**Impact of Exploitation:**
Successful exploitation can lead to:
- **Data Breach/Leakage:** An attacker can access data managed by another CAMS for HIS, leading to potential exposure of sensitive information.
- **Data Tampering/Modification:** Attackers may alter the data managed by other CAMS for HIS instances.
- **Resource Exhaustion:** An attacker can create unnecessary files on another CAMS for HIS, potentially leading to resource exhaustion and service disruption.
- **Denial of Service:** Ultimately, the attack can disable CAMS for HIS functionality on affected machines due to resource exhaustion.

**Attack Vectors:**
- **Compromised System:** The primary attack vector involves compromising a computer that has CAMS for HIS software installed.
- **Adjacent Network Access:** The attacker needs to be on the adjacent network.

**Required Attacker Capabilities/Position:**
- **Initial Access:** The attacker needs to gain access to a system running the vulnerable CAMS for HIS software.
- **Knowledge of Credentials:** Once a system is compromised, the attacker needs to be able to access and utilize stored account and password information to access other systems.
- **Adjacent Network:** The attacker must be on the same network or a network segment that is adjacent to the vulnerable system.

**Affected Products and Versions:**
The following Yokogawa products and versions are affected:
- **CENTUM CS 3000** (including CENTUM CS 3000 Small): R3.08.10 to R3.09.00 (affected if LHS4800 (CAMS for HIS) is installed)
- **CENTUM VP** (including CENTUM VP Small, CENTUM VP Basic):
    - R4.01.00 to R4.03.00 (affected only if CAMS function is used)
    - R5.01.00 to R5.04.20 (affected regardless of CAMS function usage)
    - R6.01.00 to R6.09.00 (affected regardless of CAMS function usage)
- **Exaopc**: R3.72.00 to R3.80.00 (affected if NTPF100-S6 "For CENTUM VP Support CAMS for HIS" is installed)
- **B/M9000CS**: R5.04.01 to R5.05.01
- **B/M9000 VP**: R6.01.01 to R8.03.01

**Mitigations:**
- **Upgrade/Migration:** For older systems (CENTUM CS 3000, CENTUM VP R4.01.00-R4.03.00, R5.01.00-R5.04.20), the recommended mitigation is to migrate to the latest version of CENTUM VP.
- **Patches:** For CENTUM VP (R6.01.00 to R6.09.00), upgrade to R6.09.00 and apply patch R6.09.03. For Exaopc (R3.72.00 to R3.80.00), upgrade to R3.80.00 and apply patch R3.80.01.
- **B/M9000 Update:** While B/M9000CS and VP are not directly affected, if CENTUM is installed on the same PC, it is necessary to update CENTUM and B/M9000 to the latest version.

The provided information provides more detailed technical information than the CVE description itself, including specific affected versions and detailed mitigation strategies
````

### Understanding Context

https://nvd.nist.gov/vuln/detail/CVE-2021-1045

CVE Description
>Product: AndroidVersions: Android kernelAndroid **ID**: A-195580473References: N/A

https://github.com/CyberSecAI/cve_reference_link_crawler/blob/main/data_out/CVE-2021-1045/refined/refined.md

The LLM extracted this additional context from the reference:
- **"The vulnerability exists in the Titan-M component which is a secure element"**
- **"weaknesses": "Information Disclosure (ID)"**
  
```json
{
  "vulnerability_details": {
    "root_cause": "The provided document does not detail the root cause of the vulnerability. It only indicates the vulnerability exists in the Titan-M component.",
    "weaknesses": "Information Disclosure (ID)",
    "impact": "The vulnerability allows for information disclosure.",
     "attack_vectors": "The document does not specify the attack vector. The vulnerability exists in the Titan-M component which is a secure element, so attack vectors may vary but likely involve some form of interaction with this secure element.",
    "required_capabilities": "The document does not specify required attacker capabilities or position."
  }
}
````

The CVE Reference Link is https://source.android.com/security/bulletin/pixel/2021-11-01
- https://github.com/CyberSecAI/cve_reference_link_crawler/blob/main/data_out/CVE-2021-1045/refined/combined.md
````
CVE            References     Type  Severity Component
CVE-2021-1045  A-195580473 *  ID    High     Titan-M

ID	Information disclosure
````

## Robust Processing and State Management

The crawler is designed to be both robust and idempotent - meaning it can be safely interrupted and restarted without duplicating work or losing progress. This is achieved through a file-based state management system:

1. State Tracking:
   - Processing state is tracked through the presence of specific files in each CVE directory
   - Each phase checks for completion markers before processing:
     ```
     CVE-YYYY-XXXXX/
     ├── links.txt              # Indicates Phase 1 started
     ├── secondary_links_processed.txt  # Indicates Phase 2 completed
     └── refined/
         └── refined.md         # Indicates Phase 3 completed
     ```

2. File-Based Completion Markers:
   - Phase 1 (Primary URLs): Creates `links.txt`
   - Phase 2 (Secondary URLs): Creates `secondary_links_processed.txt`
   - Phase 3 (Vulnerability Extraction): Creates `refined/refined.md`

3. Idempotent Operation:
   - Each phase checks for its completion marker before processing
   - Already processed CVEs are skipped automatically
   - Partial progress within a phase is preserved
   - Running the tool multiple times on the same data is safe

4. Resumable Processing:
   - If interrupted, the tool can be restarted safely
   - Processing resumes from the last successful state
   - No work is duplicated when restarting
   - Progress tracking is maintained per CVE and per phase

For example, if the tool is interrupted during Phase 2 while processing CVE-2021-4034:
- Phase 1 completion (presence of `links.txt`) is preserved
- Upon restart, Phase 1 is skipped
- Phase 2 resumes with remaining unprocessed secondary URLs
- Phase 3 will start fresh since `refined/refined.md` doesn't exist

This design ensures:
- No duplicate processing or content
- Safe interruption and restart
- Progress preservation
- Efficient resource usage
- Clear processing status visibility


## Notes

### Large Files
1. A git pre-commit hook is setup to prevent files greater than 20MB being committed: .pre-commit-config.yaml
2. A bash script can be run manually to move big files to tmp/big_files: big_files.sh

### Document to Text tools
[MarkItDown](https://github.com/microsoft/markitdown) is used here.
- [Docling](https://github.com/DS4SD/docling) is an alternative.

### Archive Sources
The following archive sources were considered but not implemented due to limited content availability:
- Google Cache (e.g., https://webcache.googleusercontent.com/search?q=cache:...)
- Wayback Machine

### Links that were not successfully crawled

The log file indicates what links failed to download.

1. GitHub pulls could add 'files' to get the file diff content e.g. https://github.com/haiwen/seafile-server/pull/520 -> https://github.com/haiwen/seafile-server/pull/520/files
2. Cisa.gov content is not retrieved e.g. https://www.cisa.gov/uscert/ics/advisories/icsa-22-181-03
3. HackerOne requires human verification, and signin e.g. https://hackerone.com/reports/1256967
4. Dell advisories content is not retrieved e.g. https://www.dell.com/support/kbdoc/en-us/000198780/dsa-2022-102
5. secomea content is not retrieved https://secomea.com/cybersecurity-advisory/
6. Qualcomm content is retrieved as blank e.g. https://www.qualcomm.com/company/product-security/bulletins/november-2022-bulletin 


### Spam
Some linked URLs contain spam links e.g. https://bugzilla.redhat.com/show_bug.cgi?id=2212283#c2 (referenced from https://nvd.nist.gov/vuln/detail/CVE-2023-34318) contains 100's of spam links

### Many links

Some CVEs contain many links e.g.

- CVE-2023-44487 contains ~300 direct links 
- CVE-2024-31497
- CVE-2024-3094

## Failed Links and Processing Status Tracking

The tool maintains comprehensive tracking of failed link attempts and processing status:

1. Failed Links Tracking:
   ```
   data_out/
   └── failed_urls.csv          # Central record of all failed URLs
   ```

   The failed_urls.csv contains:
   - CVE_ID: The associated CVE identifier
   - URL: The full URL that failed
   - Status: Current status of the URL
   - Failure_Reason: Specific error or failure message
   - Timestamp: When the failure occurred

   Example:
   ```csv
   CVE_ID,URL,Status,Failure_Reason,Timestamp
   CVE-2022-26267,https://github.com/JCCD/Vul/blob/main/Piwigo_12.2.0_InforMation_Disclosure.md,failed,Request failed: 404 Not Found,2025-01-09 10:06:04
   ```

2. Processing Status:
   ```
   data_out/
   └── status/
       ├── CVE-2022-26267_status.json
       └── ...
   ```

   Each CVE gets a status JSON file tracking:
   - Completion status of all three phases
   - Timestamps for each phase
   - Status of all associated URLs
   
   Example status.json:
   ```json
   {
     "cve_id": "CVE-2022-26267",
     "timestamp": "2025-01-09 11:55:04",
     "phases": {
       "primary": {
         "completed": true,
         "timestamp": "2025-01-09 10:06:04"
       },
       "secondary": {
         "completed": true,
         "timestamp": "2025-01-09 11:55:04"
       },
       "extraction": {
         "completed": false,
         "timestamp": null
       }
     },
     "urls": [
       {
         "url": "https://github.com/...",
         "status": "failed",
         "reason": "Request failed: 404 Not Found",
         "timestamp": "2025-01-09 10:06:04"
       }
     ]
   }
   ```

3. High Level Report

A high-level report can be created from these files
````
python ./src/generate_report.py

````

5. Robust Processing:
   - Failed URLs are not retried in subsequent runs
   - Processing can be safely interrupted and resumed
   - Status tracking enables monitoring of large batch jobs
   - Clear visibility into processing progress and failure points

6. Failure Categories:
   - Request failures (404, timeouts, SSL errors)
   - Authentication requirements
   - File processing errors
   - Content conversion issues

This tracking system ensures:
- No unnecessary retries of known-failed URLs
- Clear audit trail of processing attempts
- Ability to analyze failure patterns
- Progress visibility for long-running jobs



## ToDos
1. If this is done for all published CVEs, then a directory structure per https://github.com/CVEProject/cvelistV5/tree/main/cves would be more appropriate to avoid having all CVEs in one directory.



## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### Areas for Improvement

1. Content Processing:
   - Add Markdown extension to text files
   - Extract vulnerability-specific information
   - Support for additional document formats

2. Structure:
   - Consider CVEProject/cvelistV5-style organization
   - Improve handling of large files

3. Crawler Enhancement:
   - Add GitHub pull request diff support
   - Improve vendor-specific page handling
   - Add archive.org or Google cache fallback for dead links

## License

This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
https://creativecommons.org/licenses/by-sa/4.0/
