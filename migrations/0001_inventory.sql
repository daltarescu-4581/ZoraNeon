-- fridge-watcher schema.
--
-- Safe to run against an existing FridgeFriend database: everything is
-- IF NOT EXISTS / CREATE OR REPLACE.

-- ---------------------------------------------------------------------------
-- Current contents of the fridge.
-- ---------------------------------------------------------------------------
create table if not exists public.inventory_items (
    id          uuid primary key default gen_random_uuid(),
    -- Normalised (lowercased, trimmed) by the writer so upserts collapse
    -- "Oat Milk Carton" and "oat milk carton" onto one row.
    name        text        not null unique,
    category    text,
    quantity    integer     not null default 0,
    updated_at  timestamptz not null default now()
);

-- ---------------------------------------------------------------------------
-- One row per camera event we ran through the model. This is both the audit
-- log and the review queue.
-- ---------------------------------------------------------------------------
create table if not exists public.inventory_events (
    id           uuid primary key default gen_random_uuid(),
    -- Frigate's event id (or a synthesised motion-window id). UNIQUE is what
    -- makes redelivered MQTT messages idempotent rather than double-counted.
    event_id     text        not null unique,
    item_name    text,
    direction    text        check (direction in ('IN', 'OUT', 'NO_ITEM')),
    quantity     integer     not null default 1,
    confidence   real,
    reasoning    text,
    frames_path  text,
    status       text        not null
                 check (status in ('applied', 'pending_review', 'rejected')),
    created_at   timestamptz not null default now()
);

create index if not exists inventory_events_status_idx
    on public.inventory_events (status, created_at desc);
create index if not exists inventory_events_created_idx
    on public.inventory_events (created_at desc);

-- The service writes with the service-role key, which bypasses RLS. Enabling
-- RLS with no policies keeps anon/authenticated clients out until the app
-- adds its own policies.
alter table public.inventory_items  enable row level security;
alter table public.inventory_events enable row level security;

-- ---------------------------------------------------------------------------
-- Atomic "log the event and move the stock" write.
--
-- Doing this in one statement matters: the INSERT ... ON CONFLICT DO NOTHING
-- on event_id is the idempotency guard, and the quantity delta must land in
-- the same transaction as the guard. If the client did it in two round trips,
-- a crash in between would either lose the delta or let a redelivery apply it
-- twice.
-- ---------------------------------------------------------------------------
create or replace function public.apply_inventory_event(
    p_event_id    text,
    p_item_name   text,
    p_category    text,
    p_direction   text,
    p_quantity    integer,
    p_confidence  real,
    p_reasoning   text,
    p_frames_path text,
    p_status      text
) returns jsonb
language plpgsql
security definer
set search_path = public
as $$
declare
    v_delta    integer;
    v_quantity integer;
begin
    insert into public.inventory_events (
        event_id, item_name, direction, quantity,
        confidence, reasoning, frames_path, status
    )
    values (
        p_event_id, p_item_name, p_direction, coalesce(p_quantity, 1),
        p_confidence, p_reasoning, p_frames_path, p_status
    )
    on conflict (event_id) do nothing;

    if not found then
        -- Already processed this Frigate event; do not touch stock again.
        return jsonb_build_object('logged', false, 'duplicate', true, 'applied', false);
    end if;

    if p_status is distinct from 'applied' then
        return jsonb_build_object('logged', true, 'duplicate', false, 'applied', false);
    end if;

    v_delta := case when p_direction = 'OUT'
                    then -coalesce(p_quantity, 1)
                    else  coalesce(p_quantity, 1)
               end;

    insert into public.inventory_items (name, category, quantity, updated_at)
    values (p_item_name, p_category, greatest(0, v_delta), now())
    on conflict (name) do update
        set quantity   = greatest(0, inventory_items.quantity + v_delta),
            category   = coalesce(excluded.category, inventory_items.category),
            updated_at = now()
    returning quantity into v_quantity;

    return jsonb_build_object(
        'logged', true, 'duplicate', false, 'applied', true, 'quantity', v_quantity
    );
end;
$$;

grant execute on function public.apply_inventory_event(
    text, text, text, text, integer, real, text, text, text
) to service_role;
