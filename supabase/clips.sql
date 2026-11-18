-- Stream Clipper: creators to watch, streams seen, clips made. Same Supabase project as GoPeak (free plan).
-- The GitHub Actions worker uses the service key; the website goes through the 'clips' edge function
-- (HQ password). Clips go to the public bucket 'clips' so Buffer/YouTube can fetch them.

create table if not exists public.clip_sources (
  id uuid primary key default gen_random_uuid(),
  name text not null,                       -- "IShowSpeed"
  platform text not null check (platform in ('youtube', 'twitch')),
  handle text not null,                     -- youtube: @IShowSpeed  twitch: kaicenat
  enabled boolean not null default true,
  hashtags text default '',
  created_at timestamptz not null default now(),
  unique (platform, handle)
);

create table if not exists public.clip_streams (
  id uuid primary key default gen_random_uuid(),
  source_id uuid references public.clip_sources(id) on delete set null,
  url text not null unique,                 -- watch URL of the live or VOD
  title text,
  status text not null default 'queued' check (status in ('queued', 'live', 'capturing', 'processing', 'done', 'failed')),
  started_at timestamptz,                   -- stream start (for chat timestamps)
  checked_to int not null default 0,        -- seconds of the stream already scanned for moments
  runner_seen_at timestamptz,               -- last heartbeat of the capture job
  error text,
  created_at timestamptz not null default now()
);

create table if not exists public.clip_clips (
  id uuid primary key default gen_random_uuid(),
  stream_id uuid references public.clip_streams(id) on delete cascade,
  t int not null,                           -- peak second in the stream
  score numeric,
  why text[],
  title text,
  transcript text,
  duration numeric,
  layout text,
  video_url text,
  thumb_url text,
  status text not null default 'ready' check (status in ('ready', 'approved', 'scheduled', 'posted', 'skipped')),
  buffer_post_id text,
  posted_at timestamptz,
  created_at timestamptz not null default now(),
  unique (stream_id, t)
);
create index if not exists clip_clips_status on public.clip_clips (status, score desc);

alter table public.clip_sources enable row level security;
alter table public.clip_streams enable row level security;
alter table public.clip_clips enable row level security;

insert into storage.buckets (id, name, public, file_size_limit) values ('clips', 'clips', true, 52428800)
on conflict (id) do nothing;

-- Public, harmless: lets the worker's cheap "anything to do?" check run without the service key
create or replace function public.clip_work_count() returns int
language sql security definer set search_path = '' stable as $$
  select count(*)::int from public.clip_streams
  where status in ('queued', 'live') or (status = 'capturing' and runner_seen_at < now() - interval '15 minutes') $$;
grant execute on function public.clip_work_count() to anon;

-- The posting routine (Claude + Buffer connector) reads and marks clips with this key (hash only stored here)
create or replace function public.clip_post_ok(p_key text) returns boolean
language sql immutable set search_path = '' as $$
  select encode(pg_catalog.sha256(convert_to(coalesce(p_key, ''), 'UTF8')), 'hex') = 'd28863d4873af7115184071f5903425aa59751b5d005912fe6cddb52cd235373' $$;

create or replace function public.clip_next_to_post(p_key text, p_limit int default 3) returns jsonb
language plpgsql security definer set search_path = '' stable as $$
begin
  if not public.clip_post_ok(p_key) then raise exception 'not allowed'; end if;
  return coalesce((select jsonb_agg(x) from (
    select c.id, c.title, c.transcript, c.video_url, c.thumb_url, c.duration, c.score, s.title as stream_title, src.name as creator, src.hashtags
    from public.clip_clips c join public.clip_streams s on s.id = c.stream_id left join public.clip_sources src on src.id = s.source_id
    where c.status in ('approved', 'ready') and c.video_url is not null
    order by (c.status = 'approved') desc, c.score desc nulls last, c.created_at desc
    limit least(greatest(p_limit, 1), 10)) x), '[]');
end $$;

create or replace function public.clip_mark_posted(p_key text, p_id uuid, p_buffer_id text) returns void
language plpgsql security definer set search_path = '' as $$
begin
  if not public.clip_post_ok(p_key) then raise exception 'not allowed'; end if;
  update public.clip_clips set status = 'scheduled', buffer_post_id = p_buffer_id, posted_at = now() where id = p_id;
end $$;
grant execute on function public.clip_next_to_post(text, int) to anon;
grant execute on function public.clip_mark_posted(text, uuid, text) to anon;

insert into public.clip_sources (name, platform, handle, hashtags) values
  ('IShowSpeed', 'youtube', '@IShowSpeed', '#ishowspeed #speed #gta6 #gaming #funny #shorts'),
  ('Kai Cenat', 'twitch', 'kaicenat', '#kaicenat #kai #twitch #gaming #funny #shorts')
on conflict (platform, handle) do nothing;
