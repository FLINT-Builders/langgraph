/**
 * Spec for the unscoped workflow cancellation in require_issue_link.yml.
 *
 * These tests fail while the enforcement script cancels every in-progress or
 * queued run that shares the pull request head SHA. They pass once that script
 * cancels only runs that belong to the triggering pull request.
 *
 * Run:
 *   npm install --no-save --prefix /tmp/require-issue-link-test \
 *     @octokit/core@7.0.6 \
 *     @octokit/plugin-paginate-rest@14.0.0 \
 *     @octokit/plugin-rest-endpoint-methods@17.0.0
 *   REQUIRE_ISSUE_LINK_TEST_MODULES=/tmp/require-issue-link-test/node_modules \
 *     node --test .github/workflows/require_issue_link.cancel.test.mjs
 *
 * The Octokit packages are the versions locked by actions/github-script
 * 3a2844b7e9c422d3c10d287c895573f7108da1b3 (v9.0.0), which is the action the
 * workflow pins. The test executes the workflow's inline scripts the way that
 * action does (AsyncFunction with github, context, and core) and answers the
 * list-runs request with the real paginate and endpoint client.
 */

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import test from 'node:test';

const moduleRoot = process.env.REQUIRE_ISSUE_LINK_TEST_MODULES;
if (!moduleRoot) {
  throw new Error(
    'Set REQUIRE_ISSUE_LINK_TEST_MODULES to the node_modules directory from the install command in this file\'s header',
  );
}

const moduleHref = (name, entry) =>
  pathToFileURL(path.join(moduleRoot, name, entry)).href;

const { Octokit } = await import(moduleHref('@octokit/core', 'dist-src/index.js'));
const { paginateRest } = await import(
  moduleHref('@octokit/plugin-paginate-rest', 'dist-src/index.js')
);
const { restEndpointMethods } = await import(
  moduleHref('@octokit/plugin-rest-endpoint-methods', 'dist-src/index.js')
);

const workflowPath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  'require_issue_link.yml',
);

const HEAD_SHA = 'a'.repeat(40);
const OTHER_SHA = 'b'.repeat(40);
const SELF_RUN_ID = 1000;
const SAME_PR_IN_PROGRESS = 1001;
const SAME_PR_QUEUED = 1002;
const SAME_PR_IN_PROGRESS_PAGE_TWO = 1004;
const OTHER_FORK_IN_PROGRESS = 2001;
const OTHER_FORK_SAME_BRANCH_NAME = 2002;
const OTHER_FORK_QUEUED = 2005;
const OTHER_FORK_PAGE_TWO = 2004;
const SAME_REPO_VICTIM_PR = 2100;
const MAIN_PUSH_IN_PROGRESS = 3001;
const MAIN_PUSH_QUEUED = 3002;
const MAIN_PUSH_LISTS_THIS_PR_NUMBER = 3004;
const DIFFERENT_SHA = 4001;

const triggeringPullRequest = {
  number: 501,
  state: 'open',
  user: { login: 'attacker' },
  labels: [{ name: 'external' }],
  head: {
    sha: HEAD_SHA,
    ref: 'feature',
    repo: { full_name: 'attacker/langgraph', id: 999 },
  },
};

/**
 * Identity of a run that belongs to the triggering pull request.
 *
 * Fork pull_request runs often come back with pull_requests: [] (observed on
 * GET /repos/langchain-ai/langgraph/actions/runs?head_sha=bcfc3b8798ca06cd3277bb528b7abd31c0144e5b).
 * Same-repo runs can list a pull request number, and push runs can list
 * associated pull requests that did not trigger them (observed on head SHA
 * f96105368f9e9385d4c9e38bffaa457aba17f095, event push, pull_requests 6, 17, 5).
 * The head repository, head branch, and head SHA are the fields that still
 * identify the triggering pull request when pull_requests is empty, and that
 * reject a push run even when its pull_requests array contains this PR number.
 */
function belongsToTriggeringPullRequest(run) {
  const head = triggeringPullRequest.head;
  return (
    run.head_sha === head.sha &&
    run.head_branch === head.ref &&
    run.head_repository.full_name === head.repo.full_name
  );
}

function workflowRun(fields) {
  return {
    id: fields.id,
    name: fields.name,
    status: fields.status,
    event: fields.event,
    head_branch: fields.head_branch,
    head_sha: fields.head_sha,
    head_repository: fields.head_repository,
    pull_requests: fields.pull_requests ?? [],
  };
}

const attackerRepo = { full_name: 'attacker/langgraph', id: 999 };
const victimFork = { full_name: 'victim/langgraph', id: 1001 };
const baseRepo = { full_name: 'base/langgraph', id: 1 };

function buildCatalog() {
  const runs = [
    workflowRun({
      id: SELF_RUN_ID,
      name: 'Require Issue Link',
      status: 'in_progress',
      event: 'pull_request_target',
      head_branch: 'feature',
      head_sha: HEAD_SHA,
      head_repository: attackerRepo,
      pull_requests: [],
    }),
    workflowRun({
      id: SAME_PR_IN_PROGRESS,
      name: 'CI',
      status: 'in_progress',
      event: 'pull_request',
      head_branch: 'feature',
      head_sha: HEAD_SHA,
      head_repository: attackerRepo,
      pull_requests: [],
    }),
    workflowRun({
      id: SAME_PR_QUEUED,
      name: 'bench',
      status: 'queued',
      event: 'pull_request',
      head_branch: 'feature',
      head_sha: HEAD_SHA,
      head_repository: attackerRepo,
      pull_requests: [],
    }),
    workflowRun({
      id: OTHER_FORK_IN_PROGRESS,
      name: 'CI',
      status: 'in_progress',
      event: 'pull_request',
      head_branch: 'victim-branch',
      head_sha: HEAD_SHA,
      head_repository: victimFork,
      pull_requests: [],
    }),
    workflowRun({
      id: OTHER_FORK_SAME_BRANCH_NAME,
      name: 'CI',
      status: 'in_progress',
      event: 'pull_request',
      head_branch: 'feature',
      head_sha: HEAD_SHA,
      head_repository: victimFork,
      pull_requests: [],
    }),
    workflowRun({
      id: OTHER_FORK_QUEUED,
      name: 'CI',
      status: 'queued',
      event: 'pull_request',
      head_branch: 'victim-branch',
      head_sha: HEAD_SHA,
      head_repository: victimFork,
      pull_requests: [],
    }),
    workflowRun({
      id: SAME_REPO_VICTIM_PR,
      name: 'CI',
      status: 'in_progress',
      event: 'pull_request',
      head_branch: 'internal-fix',
      head_sha: HEAD_SHA,
      head_repository: baseRepo,
      pull_requests: [{ number: 900 }],
    }),
    workflowRun({
      id: MAIN_PUSH_IN_PROGRESS,
      name: 'CI',
      status: 'in_progress',
      event: 'push',
      head_branch: 'main',
      head_sha: HEAD_SHA,
      head_repository: baseRepo,
      pull_requests: [{ number: 6 }, { number: 17 }, { number: 5 }],
    }),
    workflowRun({
      id: MAIN_PUSH_QUEUED,
      name: 'baseline',
      status: 'queued',
      event: 'push',
      head_branch: 'main',
      head_sha: HEAD_SHA,
      head_repository: baseRepo,
      pull_requests: [],
    }),
    workflowRun({
      id: MAIN_PUSH_LISTS_THIS_PR_NUMBER,
      name: 'CI',
      status: 'in_progress',
      event: 'push',
      head_branch: 'main',
      head_sha: HEAD_SHA,
      head_repository: baseRepo,
      pull_requests: [{ number: triggeringPullRequest.number }],
    }),
    workflowRun({
      id: DIFFERENT_SHA,
      name: 'CI',
      status: 'in_progress',
      event: 'push',
      head_branch: 'main',
      head_sha: OTHER_SHA,
      head_repository: baseRepo,
      pull_requests: [],
    }),
  ];

  for (let index = 0; index < 100; index += 1) {
    runs.push(
      workflowRun({
        id: 8000 + index,
        name: 'CI',
        status: 'in_progress',
        event: 'pull_request',
        head_branch: `other-${index}`,
        head_sha: HEAD_SHA,
        head_repository: { full_name: 'other/langgraph', id: 3000 + index },
        pull_requests: [],
      }),
    );
  }

  runs.push(
    workflowRun({
      id: SAME_PR_IN_PROGRESS_PAGE_TWO,
      name: 'CI',
      status: 'in_progress',
      event: 'pull_request',
      head_branch: 'feature',
      head_sha: HEAD_SHA,
      head_repository: attackerRepo,
      pull_requests: [],
    }),
    workflowRun({
      id: OTHER_FORK_PAGE_TWO,
      name: 'CI',
      status: 'in_progress',
      event: 'pull_request',
      head_branch: 'victim-page-two',
      head_sha: HEAD_SHA,
      head_repository: victimFork,
      pull_requests: [],
    }),
  );

  return runs;
}

/**
 * Stand-in for GET /repos/{owner}/{repo}/actions/runs.
 *
 * Filters that the published API applies, and that were checked against
 * langchain-ai/langgraph on 2026-11-24:
 * - head_sha returns every run associated with that SHA, across events
 *   (pull_request CI and pull_request_target for bcfc3b8798ca06cd3277bb528b7abd31c0144e5b).
 * - branch matches head_branch only. It does not take a repository, so two
 *   forks with the same branch name both match.
 * - event matches the run's event.
 * - status matches the run's status.
 * No parameter scopes the list to one pull request number.
 */
function listWorkflowRuns(catalog, requestUrl) {
  const url = new URL(requestUrl);
  let rows = catalog;
  if (url.searchParams.has('head_sha')) {
    const headSha = url.searchParams.get('head_sha');
    rows = rows.filter((run) => run.head_sha === headSha);
  }
  if (url.searchParams.has('status')) {
    const status = url.searchParams.get('status');
    rows = rows.filter((run) => run.status === status);
  }
  if (url.searchParams.has('branch')) {
    const branch = url.searchParams.get('branch');
    rows = rows.filter((run) => run.head_branch === branch);
  }
  if (url.searchParams.has('event')) {
    const event = url.searchParams.get('event');
    rows = rows.filter((run) => run.event === event);
  }
  const page = Number.parseInt(url.searchParams.get('page') || '1', 10);
  const perPage = Number.parseInt(url.searchParams.get('per_page') || '30', 10);
  const start = (page - 1) * perPage;
  const workflowRuns = rows.slice(start, start + perPage);
  const headers = { 'content-type': 'application/json' };
  if (start + perPage < rows.length) {
    const next = new URL(url);
    next.searchParams.set('page', String(page + 1));
    next.searchParams.set('per_page', String(perPage));
    headers.link = `<${next.toString()}>; rel="next"`;
  }
  return {
    status: 200,
    headers,
    body: { total_count: rows.length, workflow_runs: workflowRuns },
  };
}

function jsonResponse(requestUrl, status, body, headers) {
  const response = new Response(JSON.stringify(body), { status, headers });
  Object.defineProperty(response, 'url', { value: requestUrl });
  return response;
}

function createGitHub(catalog, cancelled) {
  const OctokitWithPlugins = Octokit.plugin(paginateRest, restEndpointMethods);
  const octokit = new OctokitWithPlugins({
    auth: 'placeholder-token',
    request: {
      fetch: async (requestUrl, options) => {
        const url = new URL(requestUrl);
        const method = options.method ?? 'GET';
        const pathname = url.pathname;

        if (method === 'GET' && /\/actions\/runs$/.test(pathname)) {
          const listed = listWorkflowRuns(catalog, requestUrl);
          return jsonResponse(requestUrl, listed.status, listed.body, listed.headers);
        }

        const cancelMatch = pathname.match(/\/actions\/runs\/(\d+)\/cancel$/);
        if (method === 'POST' && cancelMatch) {
          cancelled.push(Number.parseInt(cancelMatch[1], 10));
          return jsonResponse(requestUrl, 202, {}, { 'content-type': 'application/json' });
        }

        if (method === 'GET' && /\/issues\/\d+\/labels$/.test(pathname)) {
          return jsonResponse(
            requestUrl,
            200,
            [{ name: 'external' }],
            { 'content-type': 'application/json' },
          );
        }

        if (method === 'GET' && /\/issues\/\d+\/comments$/.test(pathname)) {
          return jsonResponse(requestUrl, 200, [], { 'content-type': 'application/json' });
        }

        if (method === 'POST' && /\/issues\/\d+\/comments$/.test(pathname)) {
          return jsonResponse(requestUrl, 201, { id: 1 }, { 'content-type': 'application/json' });
        }

        if (method === 'GET' && /\/issues\/\d+$/.test(pathname)) {
          return jsonResponse(
            requestUrl,
            200,
            { number: 42, assignees: [{ login: 'maintainer' }] },
            { 'content-type': 'application/json' },
          );
        }

        if (method === 'PATCH' && /\/pulls\/\d+$/.test(pathname)) {
          return jsonResponse(requestUrl, 200, { state: 'closed' }, { 'content-type': 'application/json' });
        }

        throw new Error(`Unexpected GitHub API request ${method} ${pathname}`);
      },
    },
  });
  return octokit;
}

// actions/github-script src/async-function.ts at 3a2844b7e9c422d3c10d287c895573f7108da1b3
function callAsyncFunction(args, source) {
  const AsyncFunction = Object.getPrototypeOf(async () => null).constructor;
  const fn = new AsyncFunction(...Object.keys(args), source);
  return fn(...Object.values(args));
}

function extractLiteralBlocks(yamlText, key) {
  const lines = yamlText.split('\n');
  const blocks = [];
  const marker = new RegExp(`^(\\s*)${key}:\\s*\\|\\s*$`);
  for (let index = 0; index < lines.length; index += 1) {
    if (!marker.test(lines[index])) continue;
    const body = [];
    let blockIndent = null;
    for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
      const line = lines[cursor];
      if (line.trim() === '') {
        body.push('');
        continue;
      }
      const indent = line.match(/^\s*/)[0].length;
      if (blockIndent === null) blockIndent = indent;
      if (indent < blockIndent) break;
      body.push(line.slice(blockIndent));
    }
    while (body.length > 0 && body[body.length - 1] === '') body.pop();
    blocks.push(body.join('\n'));
  }
  return blocks;
}

function workflowScripts() {
  const yamlText = fs.readFileSync(workflowPath, 'utf8');
  const scripts = extractLiteralBlocks(yamlText, 'script');
  const checkLink = scripts.find((script) => script.includes("setOutput('has-link'"));
  const cancelScripts = scripts.filter((script) => script.includes('cancelWorkflowRun'));
  assert.ok(checkLink, 'check-link script was not found in require_issue_link.yml');
  assert.equal(cancelScripts.length, 1, 'expected one cancellation script');
  return { checkLink, cancelScript: cancelScripts[0] };
}

function substituteStepOutputs(script, outputs) {
  return script.replaceAll(/\$\{\{\s*([^}]+?)\s*\}\}/g, (match, expression) => {
    const output = expression.match(/^steps\.check-link\.outputs\.([A-Za-z0-9_-]+)$/);
    assert.ok(output, `unsubstituted Actions expression ${match}`);
    const value = outputs[output[1]];
    assert.ok(value === 'true' || value === 'false', `missing output ${output[1]}`);
    return value;
  });
}

function createCore() {
  const outputs = {};
  return {
    outputs,
    failure: undefined,
    warnings: [],
    setOutput(name, value) {
      outputs[name] = value;
    },
    setFailed(message) {
      this.failure = message;
    },
    warning(message) {
      this.warnings.push(message);
    },
  };
}

function createContext(body) {
  return {
    repo: { owner: 'base', repo: 'langgraph' },
    runId: SELF_RUN_ID,
    payload: {
      action: 'labeled',
      label: { name: 'external' },
      sender: { login: 'attacker', type: 'User' },
      pull_request: { ...triggeringPullRequest, body },
    },
  };
}

async function runEnforcement(body) {
  const { checkLink, cancelScript } = workflowScripts();
  const catalog = buildCatalog();
  const cancelled = [];
  const github = createGitHub(catalog, cancelled);
  const context = createContext(body);
  const core = createCore();
  await callAsyncFunction({ github, context, core }, checkLink);
  assert.equal(core.outputs['has-link'] === 'true' && core.outputs['is-assigned'] === 'true', false);
  const source = substituteStepOutputs(cancelScript, core.outputs);
  await callAsyncFunction({ github, context, core }, source);
  return { cancelled, failure: core.failure };
}

function assertCancellationIsScoped(cancelled) {
  const catalog = buildCatalog();
  const byId = new Map(catalog.map((run) => [run.id, run]));
  const cancelledIds = new Set(cancelled);

  assert.equal(cancelledIds.has(SELF_RUN_ID), false, 'the running require-issue-link job cancelled itself');
  assert.equal(
    cancelledIds.has(DIFFERENT_SHA),
    false,
    'a run with a different head SHA was cancelled',
  );

  for (const id of cancelledIds) {
    const run = byId.get(id);
    assert.ok(run, `cancelled unknown run ${id}`);
    assert.equal(
      belongsToTriggeringPullRequest(run),
      true,
      `cancelled run ${id} (${run.event} ${run.name} on ${run.head_repository.full_name}:${run.head_branch}) does not belong to pull request #${triggeringPullRequest.number}`,
    );
  }

  for (const run of catalog) {
    if (run.id === SELF_RUN_ID) continue;
    if (run.status !== 'in_progress' && run.status !== 'queued') continue;
    if (!belongsToTriggeringPullRequest(run)) continue;
    assert.equal(
      cancelledIds.has(run.id),
      true,
      `left run ${run.id} (${run.name}, ${run.status}) for this pull request running`,
    );
  }
}

test('a fork pull request with no issue link cancels only that pull request\'s runs', async () => {
  const { cancelled } = await runEnforcement('Ship this change.');
  assertCancellationIsScoped(cancelled);
  assert.equal(cancelled.includes(OTHER_FORK_IN_PROGRESS), false);
  assert.equal(cancelled.includes(OTHER_FORK_SAME_BRANCH_NAME), false);
  assert.equal(cancelled.includes(SAME_REPO_VICTIM_PR), false);
  assert.equal(cancelled.includes(MAIN_PUSH_IN_PROGRESS), false);
  assert.equal(cancelled.includes(MAIN_PUSH_QUEUED), false);
  assert.equal(cancelled.includes(MAIN_PUSH_LISTS_THIS_PR_NUMBER), false);
  assert.equal(cancelled.includes(OTHER_FORK_PAGE_TWO), false);
  assert.equal(cancelled.includes(SAME_PR_IN_PROGRESS), true);
  assert.equal(cancelled.includes(SAME_PR_IN_PROGRESS_PAGE_TWO), true);
  assert.equal(cancelled.includes(SAME_PR_QUEUED), true);
});

test('a fork pull request whose author is not assigned cancels only that pull request\'s runs', async () => {
  const { cancelled } = await runEnforcement('Fixes #42');
  assertCancellationIsScoped(cancelled);
  assert.equal(cancelled.includes(OTHER_FORK_QUEUED), false);
  assert.equal(cancelled.includes(MAIN_PUSH_LISTS_THIS_PR_NUMBER), false);
  assert.equal(cancelled.includes(SAME_PR_QUEUED), true);
});
